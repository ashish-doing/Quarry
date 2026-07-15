"""
mock_feed.py

Same simulated telemetry/sightings as before, now wrapped in a proper
MATCH lifecycle: lobby -> active -> ended -> (play again) -> lobby.

Phases:
  "lobby"  -- players joining, registering targets, nobody moving yet
  "active" -- robot patrols, sightings appear, voting happens, countdown running
  "ended"  -- match over, final leaderboard/bounty_log frozen for the results screen

Once Phase B/C hardware is ready, this whole module gets swapped for
real_feed.py. Keep the same message shapes documented in main.py's
module docstring so the frontend doesn't need to change.
"""

import asyncio
import math
import random
import time
import uuid
from datetime import datetime, timezone

WAYPOINTS = [
    {"id": "W1", "x": 15, "y": 20},
    {"id": "W2", "x": 50, "y": 12},
    {"id": "W3", "x": 85, "y": 25},
    {"id": "W4", "x": 82, "y": 65},
    {"id": "W5", "x": 48, "y": 80},
    {"id": "W6", "x": 14, "y": 62},
]

DETECTABLE_LABELS = ["person", "backpack", "toolbox", "chair", "unknown object"]
CONFIRM_THRESHOLD = 2
DEFAULT_DURATION = 180  # seconds


class Candidate:
    def __init__(self, label, confidence, waypoint, is_registered_target):
        self.id = str(uuid.uuid4())[:8]
        self.label = label
        self.confidence = confidence
        self.waypoint = waypoint
        self.is_registered_target = is_registered_target
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.votes = {}
        self.status = "pending"

    def net_confirms(self):
        confirms = sum(1 for v in self.votes.values() if v == "confirm")
        disputes = sum(1 for v in self.votes.values() if v == "dispute")
        return confirms - disputes

    def vote_tally(self):
        confirms = sum(1 for v in self.votes.values() if v == "confirm")
        disputes = sum(1 for v in self.votes.values() if v == "dispute")
        return confirms, disputes

    def to_dict(self):
        return {
            "type": "candidate",
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "waypoint": self.waypoint,
            "is_registered_target": self.is_registered_target,
            "timestamp": self.timestamp,
        }


class MockRobotState:
    def __init__(self):
        self.phase = "lobby"
        self.match_duration = DEFAULT_DURATION
        self.match_start_monotonic = None
        self.match_start_iso = None

        self._reset_round_state()

    def _reset_round_state(self):
        self.current_wp_index = 0
        self.next_wp_index = 1
        self.progress = 0.0
        self.battery = 100.0
        self.fps = 11.0
        self.registered_targets = []
        self.candidates = {}
        self.bounty_log = []
        self.leaderboard = {}

    # -- match lifecycle -------------------------------------------------
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

    # -- movement ---------------------------------------------------------
    def step(self):
        self.progress += 0.05
        if self.progress >= 1.0:
            self.progress = 0.0
            self.current_wp_index = self.next_wp_index
            self.next_wp_index = (self.next_wp_index + 1) % len(WAYPOINTS)
        self.battery = max(20.0, self.battery - random.uniform(0, 0.03))
        self.fps = round(random.uniform(8.5, 13.5), 1)

    def position(self):
        a = WAYPOINTS[self.current_wp_index]
        b = WAYPOINTS[self.next_wp_index]
        x = a["x"] + (b["x"] - a["x"]) * self.progress
        y = a["y"] + (b["y"] - a["y"]) * self.progress
        return {"x": round(x, 2), "y": round(y, 2)}

    def heading(self):
        a = WAYPOINTS[self.current_wp_index]
        b = WAYPOINTS[self.next_wp_index]
        angle = math.degrees(math.atan2(b["x"] - a["x"], -(b["y"] - a["y"])))
        return round(angle, 1)

    def speed(self):
        return round(random.uniform(0.15, 0.35), 2)

    def current_waypoint_id(self):
        if self.progress < 0.08:
            return WAYPOINTS[self.current_wp_index]["id"]
        return None

    # -- sightings ----------------------------------------------------------
    def maybe_raise_candidate(self):
        wp_id = self.current_waypoint_id()
        if not wp_id or random.random() > 0.30:
            return None

        if self.registered_targets and random.random() < 0.45:
            target = random.choice(self.registered_targets)
            label, is_registered = target["name"], True
            confidence = round(random.uniform(0.5, 0.9), 2)
        else:
            label, is_registered = random.choice(DETECTABLE_LABELS), False
            confidence = round(random.uniform(0.35, 0.8), 2)

        candidate = Candidate(label, confidence, wp_id, is_registered)
        self.candidates[candidate.id] = candidate
        return candidate

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

    def register_target(self, name, note=""):
        self.registered_targets.append({"name": name, "note": note})

    def leaderboard_sorted(self):
        return sorted(
            [{"name": n, "score": s} for n, s in self.leaderboard.items()],
            key=lambda e: -e["score"],
        )


state = MockRobotState()


async def telemetry_loop(broadcast):
    while True:
        if state.phase == "active":
            state.step()
            telemetry = {
                "type": "telemetry",
                "position": state.position(),
                "heading": state.heading(),
                "speed": state.speed(),
                "waypoint": state.current_waypoint_id(),
                "battery": round(state.battery, 1),
                "fps": state.fps,
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

        await asyncio.sleep(0.6)