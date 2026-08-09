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
same `candidate` events mock_feed used to fabricate randomly. Shared
by BOTH the inner UGV (inner_agent.py) and outer UGVs (site_agent.py)
-- this module doesn't know or care which role is using it; the
distinction lives one layer up.

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
import math
import os
import time
from datetime import datetime, timezone
from typing import Optional

from collections import deque

import numpy as np

import cyberwave
from backend.mock_feed import Candidate as _BaseCandidate
from backend.vision.detector import YoloeDetector
from backend.vision.reid import TargetMatcher
from backend.vision.collision_guard import CollisionGuard
from backend.vision.ocr_check import MarkerOCR
from backend.objects_registry import load_objects, expected_number_for
from backend.nav_math import location_offset as _nav_location_offset

logger = logging.getLogger("quarry.real_feed")

# ---------------------------------------------------------------------------
# Tag-to-waypoint mapping is now fixed (see apriltag_detector.py): tags
# 0-5 map 1:1 to W1-W6, tags 6-7 are reserved as fixed origin/reference
# markers used to establish the room's coordinate frame -- NOT waypoints
# themselves. x/y below are STILL PLACEHOLDER until measure_waypoints.py
# is run at the actual venue and the recorded tag positions are converted
# into this same (x, y) frame. Keep the same units/frame as get_pose()
# returns -- do not just paste measure_waypoints.py's raw output here,
# that's in the CAMERA's frame relative to two reference tags, not
# get_pose()'s frame. The conversion between those two frames is real
# work that hasn't been done yet; measure_waypoints.py's README section
# explains why it can't do that conversion for you automatically.
# ---------------------------------------------------------------------------
WAYPOINTS = [
    {"id": "W1", "tag_id": 0, "x": 15, "y": 20},   # still placeholder -- see note above
    {"id": "W2", "tag_id": 1, "x": 50, "y": 12},
    {"id": "W3", "tag_id": 2, "x": 85, "y": 25},
    {"id": "W4", "tag_id": 3, "x": 82, "y": 65},
    {"id": "W5", "tag_id": 4, "x": 48, "y": 80},
    {"id": "W6", "tag_id": 5, "x": 14, "y": 62},
]
REFERENCE_TAG_IDS = (6, 7)  # fixed origin markers, not patrol waypoints -- place these FIRST at venue
WAYPOINT_SNAP_RADIUS = 9999  # TEMP: testing detection only, ignore waypoint gating for now

ENVIRONMENT_ID = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
ASSET_KEY = "waveshare/ugv-beast"
TWIN_NAME = "QUARRY"

DETECTION_COOLDOWN_S = 25.0  # don't re-raise the same label+waypoint faster than this --
                              # raised from 4.0: at 4s, a stationary object generated a
                              # fresh candidate every cooldown window regardless of whether
                              # the previous one had been voted on yet, flooding the
                              # sightings panel (confirmed: 100+ in a 2-3 min session)
CONFIRM_THRESHOLD = 2
DEFAULT_DURATION = 180
TELEMETRY_TICK_S = 0.2       # tighter than mock_feed's 0.6s -- real vision needs it
TWIN_SYNC_INTERVAL_S = 2.0   # how often to push position to the Cyberwave platform's own
                              # digital-twin view via set_pose(). NOTE: set_pose()'s own
                              # docstring says "stub in PR3" as of this SDK version -- this
                              # is wired and called, but VERIFY in the Cyberwave platform UI
                              # that the twin actually visibly moves before claiming this
                              # works anywhere. If it doesn't move, this is a documented
                              # attempt against a real method, not a confirmed working feature

SMOOTHING_WINDOW = 5          # consider the last 5 detection ticks at a waypoint
SMOOTHING_MIN_HITS = 3        # a label must appear in >=3 of the last 5 to be trusted --
                               # a lightweight, ByteTrack-family stand-in for a real tracker;
                               # single-frame detections are a fluke risk, don't raise on one
CONFIDENCE_EMA_ALPHA = 0.35   # exponential-moving-average weight for the rolling confidence shown
                               # alongside the raw single-frame number in the sightings panel

# -- capture framing gate ----------------------------------------------------
# Per the mission requirement: a proof photo should be a properly-framed
# capture, not whatever happened to be in view the instant the smoothing
# window cleared. This is a MINIMUM bbox-area-fraction gate -- too small
# means the object is still far/at the frame edge, so candidate-raising
# (and therefore the proof photo) waits rather than firing on a distant,
# hard-to-verify sighting. This is deliberately independent from
# CollisionGuard's proximity threshold (which is a MAXIMUM, for safety) --
# same underlying bbox-area-as-proxy technique, opposite purpose, not the
# same constant, don't conflate the two.
MIN_FRAME_FRACTION = 0.05

# -- intruder detection -------------------------------------------------
# Any unregistered "person" detection is treated as a possible intruder,
# checked independently of the numbered-object candidate pipeline (no
# waypoint gating, no multi-tick smoothing requirement -- an intruder
# alert needs to be fast and cheap to fire, not de-duplicated the way a
# numbered-object sighting is, since false-alarm cost here is much lower
# than a missed real one). Deliberately NOT cross-checked against
# registered targets -- registered targets in this mission are numbered
# props (boxes, etc.), not people, so there's no "known person" case to
# exclude.
INTRUDER_LABEL = "person"
INTRUDER_COOLDOWN_S = 15.0


def _location_offset_for(waypoint_id: Optional[str], position: dict) -> Optional[dict]:
    """Waypoint + offset, per the master doc's locked location decision --
    a cheap vector subtraction against the object's nearest known waypoint,
    no live AprilTag dependency at runtime. Returns None if waypoint_id
    doesn't resolve (e.g. robot isn't currently snapped to any waypoint).
    Thin wrapper around nav_math.location_offset() -- the actual math lives
    there (single source of truth, unit-tested in test_nav_math.py)."""
    if waypoint_id is None:
        return None
    return _nav_location_offset(waypoint_id, position, WAYPOINTS)


class Candidate(_BaseCandidate):
    """Extends mock_feed.Candidate with instant_confidence, location_offset,
    and OCR cross-check fields -- purely additive to the WebSocket contract.
    `confidence` (inherited) is now the EMA-smoothed rolling value used for
    gating/scoring; `instant_confidence` is the raw single-frame reading,
    kept only for display. `location_offset` is the waypoint+offset location
    (Section 5 Task 6). `ocr_number`/`ocr_anomaly` are the OCR cross-check
    result (Section 4's "OSNet primary, OCR cross-check" rule) -- ocr_anomaly
    is None unless OCR actually disagreed with the OSNet match. Note:
    `image`/`twin_semantic_label` (the mission-narrative additions for the
    proof gallery and identity-vs-twin comparison) are deliberately NOT
    fields on this class -- they're attached by the relay layer
    (inner_agent.py / site_agent.py) at send time, since this class's job
    is CV/telemetry state, not WebSocket payload shaping."""

    def __init__(self, label, confidence, waypoint, is_registered_target, instant_confidence=None,
                 location_offset=None, ocr_number=None, ocr_anomaly=None):
        super().__init__(label, confidence, waypoint, is_registered_target)
        self.instant_confidence = confidence if instant_confidence is None else instant_confidence
        self.location_offset = location_offset
        self.ocr_number = ocr_number
        self.ocr_anomaly = ocr_anomaly

    def to_dict(self):
        d = super().to_dict()
        d["instant_confidence"] = self.instant_confidence
        d["location_offset"] = self.location_offset
        d["ocr_number"] = self.ocr_number
        d["ocr_anomaly"] = self.ocr_anomaly
        return d


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

        # OCR cross-check + sequential-object registry -- both additive,
        # non-fatal if unavailable (ocr.available gates all use of it below;
        # an empty objects registry just means no expected-number lookups
        # ever succeed, matching falls back to "no anomaly check possible").
        self.ocr = MarkerOCR()
        self.objects_registry = load_objects()

        self.registered_targets = []
        self.candidates = {}
        self.bounty_log = []
        self.leaderboard = {}
        self._last_candidate_at = {}  # (label, waypoint) -> monotonic time, cooldown tracking
        self._last_intruder_alert_at = 0.0
        self._label_history: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._rolling_confidence: dict = {}  # label -> EMA confidence

        self._position = {"x": 0.0, "y": 0.0}
        self._heading = 0.0
        self._speed = 0.0
        self._battery = 100.0
        self._fps = 0.0
        self._last_twin_sync = 0.0

    def _reset_round_state(self):
        self.registered_targets = []
        self.candidates = {}
        self.bounty_log = []
        self.leaderboard = {}
        self._last_candidate_at = {}
        self._label_history.clear()
        self._rolling_confidence = {}

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

        # Push position to Cyberwave's own digital-twin view. Confirmed via
        # `python -m backend.diagnose_twin` that set_pose(x=, y=, ...) is a
        # real method on the twin -- yaw is deliberately omitted, same reason
        # heading is never guessed elsewhere in this file. Throttled to avoid
        # flooding MQTT with a write every 0.2s tick.
        now = time.monotonic()
        if now - self._last_twin_sync >= TWIN_SYNC_INTERVAL_S:
            self._last_twin_sync = now
            try:
                self.twin.set_pose(x=self._position["x"], y=self._position["y"])
            except Exception as exc:  # noqa: BLE001 -- a failed sync must not break telemetry
                logger.debug("twin.set_pose failed (%s) -- platform twin view may be stale", exc)

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
            return self.twin.get_frame('numpy', source='remote_edge')
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_frame failed (%s)", exc)
            return None

    @staticmethod
    def _crop(frame: np.ndarray, bbox) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        return frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]

    def _update_smoothing(self, detections):
        """Feed one tick's detections into the rolling tracker. Returns
        (stable_label, best_detection) once that label has cleared the
        majority-vote threshold across the recent window, else (None, None)
        -- meaning "seen, but not consistently enough yet to trust."""
        best = max(detections, key=lambda d: d.confidence) if detections else None
        self._label_history.append(best.label if best else None)

        if best is None:
            return None, None

        prev = self._rolling_confidence.get(best.label, best.confidence)
        self._rolling_confidence[best.label] = (
            CONFIDENCE_EMA_ALPHA * best.confidence + (1 - CONFIDENCE_EMA_ALPHA) * prev
        )

        hits = sum(1 for lbl in self._label_history if lbl == best.label)
        if hits >= SMOOTHING_MIN_HITS:
            return best.label, best
        return None, None

    def check_intruder(self, detections) -> Optional[dict]:
        """Independent of the numbered-object candidate pipeline below --
        run this every tick alongside maybe_raise_candidate() using the
        SAME detections list (no extra detect() call). Returns an
        alert-shaped dict ready to hand to the relay layer's
        alert_report message, or None if nothing worth alerting this
        tick. See the module-level INTRUDER_* constants' comment for why
        this doesn't do waypoint/cooldown gating the same way candidates
        do."""
        if not detections:
            return None
        now = time.monotonic()
        if now - self._last_intruder_alert_at < INTRUDER_COOLDOWN_S:
            return None

        person = max(
            (d for d in detections if d.label == INTRUDER_LABEL),
            key=lambda d: d.confidence, default=None,
        )
        if person is None:
            return None

        self._last_intruder_alert_at = now
        wp_id = self.current_waypoint_id()
        distance_m = None
        for wp in WAYPOINTS:
            if wp["id"] == wp_id:
                distance_m = round(math.hypot(self._position["x"] - wp["x"], self._position["y"] - wp["y"]), 3)
                break

        return {
            "kind": "intruder",
            "position": dict(self._position),
            "nearest_waypoint": wp_id,
            "distance_m": distance_m,
            "message": f"Possible intruder detected (confidence {person.confidence:.2f}).",
        }

    def maybe_raise_candidate(self, frame=None, detections=None) -> Optional[Candidate]:
        wp_id = self.current_waypoint_id()
        if not wp_id or not self.detector.available:
            return None

        # Accept a pre-captured frame + pre-computed detections so callers
        # (site_agent.py / inner_agent.py's video overlay) can share one
        # capture+detect per tick instead of each doing their own -- was a
        # real, flagged redundancy: multiple independent frame pulls per
        # tick otherwise.
        if frame is None or detections is None:
            frame = self._capture_frame()
            if frame is None:
                return None
            detections = self.detector.detect(frame)

        stable_label, best = self._update_smoothing(detections)
        if stable_label is None:
            return None  # nothing seen, or seen but not consistently enough yet

        # Capture framing gate: a properly-framed proof photo means the
        # object isn't still tiny/at-the-edge of frame. Below this
        # threshold, wait rather than raise -- the smoothing window can
        # clear on a distant, barely-visible detection just as easily as
        # a close, well-framed one, and this mission explicitly wants the
        # proof photo to actually be a good capture.
        if CollisionGuard._bbox_area_fraction(best.bbox) < MIN_FRAME_FRACTION:
            return None

        # Don't raise a new candidate if one for this exact label+waypoint is
        # already open and awaiting a vote -- the cooldown alone wasn't enough,
        # since it only throttled *rate*, not duplicates. A pending sighting
        # should get confirmed/disputed before a fresh one for the same object
        # appears, otherwise a stationary target floods the panel indefinitely.
        already_pending = any(
            c.label == stable_label and c.waypoint == wp_id and c.status == "pending"
            for c in self.candidates.values()
        )
        if already_pending:
            return None

        key = (stable_label, wp_id)
        now = time.monotonic()
        if now - self._last_candidate_at.get(key, 0) < DETECTION_COOLDOWN_S:
            return None  # cooldown -- don't spam a candidate every single frame
        self._last_candidate_at[key] = now


        # Crop is now taken unconditionally, not just when matching against
        # registered targets -- this is the raw material for the proof
        # gallery / data flywheel. Matching logic itself is unchanged.
        crop = self._crop(frame, best.bbox)

        label, is_registered = stable_label, False
        if self.matcher.available and self.registered_targets:
            match_name, _score = self.matcher.best_match(crop)
            if match_name:
                label, is_registered = match_name, True

        rolling_conf = round(self._rolling_confidence.get(stable_label, best.confidence), 2)

        # Waypoint + offset location (Section 5 Task 6) -- cheap vector
        # subtraction against the object's nearest known waypoint, computed
        # unconditionally since it costs nothing and every candidate benefits.
        location_offset = _location_offset_for(wp_id, self._position)

        # OCR cross-check (Section 4's locked "OSNet primary, OCR
        # cross-check" rule) -- only meaningful for a registered target with
        # a known expected number, and only if Tesseract is actually
        # available. A missing OCR signal (no expected number, OCR
        # unavailable, or nothing readable this frame) is never itself an
        # anomaly -- only a genuine number MISMATCH is.
        ocr_number, ocr_anomaly = None, None
        if is_registered and self.ocr.available:
            expected_number = expected_number_for(label, self.objects_registry)
            if expected_number is not None:
                ocr_result = self.ocr.read_number(crop)
                ocr_number = ocr_result.number
                ocr_anomaly = self.ocr.cross_check(ocr_result, expected_number)

        candidate = Candidate(
            label, rolling_conf, wp_id, is_registered,
            instant_confidence=round(best.confidence, 2),
            location_offset=location_offset,
            ocr_number=ocr_number,
            ocr_anomaly=ocr_anomaly,
        )
        # training_crop is a same-process, local handoff to the relay layer
        # (site_agent.py / inner_agent.py) -- NOT part of Candidate.to_dict().
        # The relay layer decides whether/how to encode it for transmission
        # (the mission's proof-gallery `image` field is built there, not here).
        candidate.training_crop = crop
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
            "location_offset": candidate.location_offset,
            "ocr_number": candidate.ocr_number,
            "ocr_anomaly": candidate.ocr_anomaly,
        }
        self.bounty_log.insert(0, entry)
        self.bounty_log = self.bounty_log[:50]
        for name in contributors:
            self.leaderboard[name] = self.leaderboard.get(name, 0) + 1
        del self.candidates[candidate.id]

    def register_target(self, name: str, note: str = "", photos: Optional[list] = None):
        """photos, if given, must be a list of already-decoded HxWx3 numpy
        arrays (decode uploads with cv2.imdecode before calling this).
        Register from 3-5 angles for real re-id robustness -- a single
        photo still works but is the weaker case. In the current mission
        flow this is called once, offline, before a run starts (no more
        live in-session registration UI) -- see objects_registry.py and
        register_mission_objects.py."""
        self.registered_targets.append({"name": name, "note": note})
        if photos:
            self.matcher.register(name, photos)
        else:
            logger.warning(
                "Target '%s' registered without any photos -- it can only be confirmed "
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