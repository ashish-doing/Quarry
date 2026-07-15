"""
real_feed.py -- live hardware replacement for mock_feed.py.

Matches mock_feed.py's public surface exactly (state, WAYPOINTS,
telemetry_loop, and reuses the Candidate class) so main.py's
    from backend import mock_feed
becomes
    from backend import real_feed as mock_feed
with no other changes -- see the WebSocket contract documented in
main.py's module docstring, which this module honors precisely.

Pulls telemetry + camera frames from the QUARRY twin via the
Cyberwave SDK, runs YOLOE-26 detection + OSNet re-id, and raises the
same `candidate` events mock_feed used to fabricate randomly.

REQUIRED BEFORE FIRST RUN:
  1. Set CYBERWAVE_API_KEY in your environment (or .env, loaded by main.py).
  2. Update WAYPOINTS below with real-world coordinates once AprilTags
     are physically placed and measured (master doc Section 7, item 2).
  3. Confirm the units/frame WAYPOINTS uses match whatever get_pose()
     actually returns -- print a few live poses first and sanity check
     against a tape measure before trusting waypoint snapping.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np

import cyberwave
from backend.mock_feed import Candidate  # reuse the exact same shape/contract
from backend.vision.detector import YoloeDetector
from backend.vision.reid import TargetMatcher
from backend.vision.collision_guard import CollisionGuard

logger = logging.getLogger("quarry.real_feed")

# ---------------------------------------------------------------------------
# PLACEHOLDER -- replace with real measured coordinates once AprilTags are
# placed. Keep the same units/frame as get_pose() returns, and the same
# frame the frontend's Three.js map already expects (see map3d.js).
# ---------------------------------------------------------------------------
WAYPOINTS = [
    {"id": "W1", "x": 15, "y": 20},
    {"id": "W2", "x": 50, "y": 12},
    {"id": "W3", "x": 85, "y": 25},
    {"id": "W4", "x": 82, "y": 65},
    {"id": "W5", "x": 48, "y": 80},
    {"id": "W6", "x": 14, "y": 62},
]
WAYPOINT_SNAP_RADIUS = 9999  # TEMP: testing detection only, ignore waypoint gating for now

ENVIRONMENT_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
ASSET_KEY = "waveshare/ugv-beast"
TWIN_NAME = "QUARRY"

DETECTION_COOLDOWN_S = 4.0   # don't re-raise the same label+waypoint faster than this
CONFIRM_THRESHOLD = 2
DEFAULT_DURATION = 180
TELEMETRY_TICK_S = 0.2       # tighter than mock_feed's 0.6s -- real vision needs it


def _nearest_waypoint(x: float, y: float) -> Optional[str]:
    best_id, best_dist = None, WAYPOINT_SNAP_RADIUS
    for wp in WAYPOINTS:
        d = ((wp["x"] - x) ** 2 + (wp["y"] - y) ** 2) ** 0.5
        if d < best_dist:
            best_id, best_dist = wp["id"], d
    return best_id


class RealRobotState:
    """Mirrors mock_feed.MockRobotState's public surface, backed by
    real telemetry + vision instead of random generation."""

    def __init__(self):
        self.phase = "lobby"
        self.match_duration = DEFAULT_DURATION
        self.match_start_monotonic = None
        self.match_start_iso = None

        self.twin = None
        self.detector = YoloeDetector()
        self.matcher = TargetMatcher()
        self.guard: Optional[CollisionGuard] = None  # created once twin is connected

        self.registered_targets = []
        self.candidates = {}
        self.bounty_log = []
        self.leaderboard = {}
        self._last_candidate_at = {}  # (label, waypoint) -> monotonic time, cooldown tracking

        self._position = {"x": 0.0, "y": 0.0}
        self._heading = 0.0
        self._speed = 0.0
        self._battery = 100.0
        self._fps = 0.0

    def _reset_round_state(self):
        self.registered_targets = []
        self.candidates = {}
        self.bounty_log = []
        self.leaderboard = {}
        self._last_candidate_at = {}

    def connect(self):
        if "CYBERWAVE_API_KEY" not in os.environ:
            raise RuntimeError(
                "CYBERWAVE_API_KEY not set -- add it to backend/.env "
                "(main.py already calls load_dotenv() at import time)."
            )
        cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
        self.twin = cyberwave.twin(ASSET_KEY, environment=ENVIRONMENT_ID, name=TWIN_NAME)
        logger.info("real_feed connected to twin '%s' in environment %s", TWIN_NAME, ENVIRONMENT_ID)

    # -- match lifecycle (identical semantics to mock_feed) -------------------
    def start_match(self, duration=None):
        self._reset_round_state()
        self.phase = "active"
        self.match_duration = duration or DEFAULT_DURATION
        self.match_start_monotonic = time.monotonic()
        self.match_start_iso = datetime.now(timezone.utc).isoformat()

    def remaining_seconds(self):
        if self.phase != "active" or self.match_start_monotonic is None:
            return 0
        elapsed = time.monotonic() - self.match_start_monotonic
        return max(0, round(self.match_duration - elapsed))

    def end_match(self):
        self.phase = "ended"

    def return_to_lobby(self):
        self.phase = "lobby"

    # -- telemetry --------------------------------------------------------
    def _refresh_telemetry(self):
        if self.twin is None:
            return
        try:
            pose = self.twin.get_pose()
            pos = pose.get("position", {})
            self._position = {"x": round(pos.get("x", 0.0), 2), "y": round(pos.get("y", 0.0), 2)}
            # rotation is a quaternion, not yaw degrees -- real heading conversion
            # is a follow-up item, not guessed here after getting pose wrong once already
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_pose failed (%s) -- keeping last known position", exc)

        # NOTE: t.telemetry is a publish-only interface (driver_info, publish,
        # session_start/end) -- not a readable battery/fps snapshot. Battery
        # and FPS stay unavailable until we find the real consumer-side method
        # (likely t.subscribe_position()/listen() per dir(t) -- untested).

    def position(self):
        return self._position

    def heading(self):
        return self._heading

    def speed(self):
        return self._speed

    def current_waypoint_id(self):
        return _nearest_waypoint(self._position["x"], self._position["y"])

    # -- vision -------------------------------------------------------------
    def _capture_frame(self) -> Optional[np.ndarray]:
        if self.twin is None:
            return None
        try:
            return self.twin.get_latest_frame()
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_latest_frame failed (%s)", exc)
            return None

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def maybe_raise_candidate(self) -> Optional[Candidate]:
        wp_id = self.current_waypoint_id()
        if not wp_id or not self.detector.available:
            return None

        frame = self._capture_frame()
        if frame is None:
            return None

        detections = self.detector.detect(frame)
        if not detections:
            return None

        best = max(detections, key=lambda d: d.confidence)

        key = (best.label, wp_id)
        now = time.monotonic()
        if now - self._last_candidate_at.get(key, 0) < DETECTION_COOLDOWN_S:
            return None  # cooldown -- don't spam a candidate every single frame
        self._last_candidate_at[key] = now

        label, is_registered = best.label, False
        if self.matcher.available and self.registered_targets:
            crop = self._crop(frame, best.bbox)
            match_name, _score = self.matcher.best_match(crop)
            if match_name:
                label, is_registered = match_name, True

        candidate = Candidate(label, round(best.confidence, 2), wp_id, is_registered)
        self.candidates[candidate.id] = candidate
        return candidate

    # -- voting / targets (identical logic to mock_feed) ---------------------
    def cast_vote(self, candidate_id, player_name, vote):
        candidate = self.candidates.get(candidate_id)
        if not candidate or candidate.status != "pending":
            return None, False
        candidate.votes[player_name] = vote
        if candidate.net_confirms() >= CONFIRM_THRESHOLD:
            candidate.status = "confirmed"
            self._finalize(candidate)
            return candidate, True
        return candidate, False

    def _finalize(self, candidate):
        contributors = [p for p, v in candidate.votes.items() if v == "confirm"]
        entry = {
            "id": candidate.id, "label": candidate.label, "confidence": candidate.confidence,
            "waypoint": candidate.waypoint, "timestamp": candidate.timestamp,
            "contributors": contributors,
        }
        self.bounty_log.insert(0, entry)
        self.bounty_log = self.bounty_log[:50]
        for name in contributors:
            self.leaderboard[name] = self.leaderboard.get(name, 0) + 1
        del self.candidates[candidate.id]

    def register_target(self, name: str, note: str = "", photo: Optional[np.ndarray] = None):
        """photo, if given, must already be a decoded HxWx3 numpy array
        (see the main.py patch note -- decode the upload with cv2.imdecode
        before calling this)."""
        self.registered_targets.append({"name": name, "note": note})
        if photo is not None:
            self.matcher.register(name, photo)
        else:
            logger.warning(
                "Target '%s' registered without a photo -- it can only be confirmed "
                "by manual team voting, never auto-matched by appearance.", name,
            )

    def leaderboard_sorted(self):
        return sorted(
            [{"name": n, "score": s} for n, s in self.leaderboard.items()],
            key=lambda e: -e["score"],
        )


state = RealRobotState()


async def telemetry_loop(broadcast):
    state.connect()

    # Collision guard runs independently of the match-phase gating below --
    # it must watch for obstacles any time the robot could be moving,
    # including manual teleop outside an active match. New message types
    # ("collision_alert"/"collision_clear") are additive to the contract:
    # existing frontend code that doesn't know about them will just ignore
    # them, same as any unrecognized WebSocket message type.
    def _emit_collision_event(event: dict):
        asyncio.create_task(broadcast(event))

    state.guard = CollisionGuard(state.twin, state.detector, on_event=_emit_collision_event)
    asyncio.create_task(state.guard.run())

    while True:
        if state.phase == "active":
            state._refresh_telemetry()
            telemetry = {
                "type": "telemetry",
                "position": state.position(),
                "heading": state.heading(),
                "speed": state.speed(),
                "waypoint": state.current_waypoint_id(),
                "battery": state._battery,
                "fps": state._fps,
                "remaining_seconds": state.remaining_seconds(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await broadcast(telemetry)

            candidate = state.maybe_raise_candidate()
            if candidate:
                await broadcast(candidate.to_dict())

            if state.remaining_seconds() <= 0:
                state.end_match()
                await broadcast({
                    "type": "match_ended",
                    "leaderboard": state.leaderboard_sorted(),
                    "bounty_log": state.bounty_log,
                    "duration": state.match_duration,
                })

        await asyncio.sleep(TELEMETRY_TICK_S)