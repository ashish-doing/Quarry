"""
test_dispute_and_safety_edge_cases.py -- fills two real gaps flagged in
FMEA.md: the dispute-resolution path (symmetric to confirm, added later
than confirm and easy to leave under-tested) and CollisionGuard's
debounce/re-arm logic (the actual safety-critical state machine, not
just the proximity-threshold happy path).

Run:
    python -m pytest backend/test_dispute_and_safety_edge_cases.py -v
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.main import app, hub, CONFIRM_THRESHOLD
from backend.vision.collision_guard import (
    CollisionGuard,
    PROXIMITY_AREA_THRESHOLD,
    CLEAR_STREAK_REQUIRED,
)


# ---------------------------------------------------------------------------
# Dispute-resolution path (Hub) -- this is the mirror of the confirm path,
# added specifically because it was previously missing entirely (a disputed
# candidate just sat "pending" forever, per main.py's own inline comment).
# ---------------------------------------------------------------------------

class FakeDetection:
    def __init__(self, bbox):
        self.bbox = bbox
        self.label = "person"
        self.confidence = 0.9


def _join(ws, name, role="field_agent", team="Alpha", location="Test Room"):
    payload = {"type": "join", "name": name, "role": role, "team": team}
    if role == "field_agent":
        payload["location"] = location
    ws.send_json(payload)
    return ws.receive_json()  # the "init" message


def test_dispute_resolution_reaches_threshold_and_rejects():
    """A candidate voted down past -CONFIRM_THRESHOLD must resolve to
    match_rejected, not sit pending forever. Mirrors the confirm-path
    test but was the actual bug this feature fixed -- confirm alone
    passing doesn't prove dispute does."""
    hub.sites.clear()
    hub.players.clear()

    client = TestClient(app)
    with client.websocket_connect("/ws") as agent_ws, \
         client.websocket_connect("/ws") as voter1_ws, \
         client.websocket_connect("/ws") as voter2_ws:

        _join(agent_ws, "AgentDispute", role="field_agent")
        _join(voter1_ws, "Voter1", role="spectator")
        _join(voter2_ws, "Voter2", role="spectator")

        agent_ws.send_json({
            "type": "candidate_report", "label": "person", "confidence": 0.8,
            "instant_confidence": 0.8, "waypoint": "W1", "is_registered_target": False,
        })

        # Drain the candidate broadcast on each socket until we see it
        candidate_id = None
        for ws in (agent_ws, voter1_ws, voter2_ws):
            while True:
                msg = ws.receive_json()
                if msg.get("type") == "candidate":
                    candidate_id = msg["id"]
                    site_id = msg["site_id"]
                    break

        assert candidate_id is not None

        # Two disputes should hit -CONFIRM_THRESHOLD (default 2)
        voter1_ws.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "dispute"})
        voter2_ws.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "dispute"})

        # Look for match_rejected on the voter's own socket within the
        # next few broadcasts (vote_update messages come first)
        seen_rejected = False
        for _ in range(8):
            msg = voter2_ws.receive_json()
            if msg.get("type") == "match_rejected":
                seen_rejected = True
                assert msg["id"] == candidate_id
                assert msg["site_id"] == site_id
                break
        assert seen_rejected, "match_rejected was never broadcast after reaching the dispute threshold"

        # Candidate must actually be gone from the site's pending set,
        # not just have fired the event -- otherwise it's a phantom
        # "pending forever" bug wearing a broadcast as a costume.
        assert candidate_id not in hub.sites[site_id].candidates


def test_disputed_candidate_not_double_counted_toward_leaderboard():
    """A rejected candidate must never contribute points -- confirms and
    disputes are mutually exclusive outcomes for the same candidate."""
    hub.sites.clear()
    hub.players.clear()

    client = TestClient(app)
    with client.websocket_connect("/ws") as agent_ws, \
         client.websocket_connect("/ws") as voter1_ws, \
         client.websocket_connect("/ws") as voter2_ws:

        _join(agent_ws, "AgentDispute2", role="field_agent")
        _join(voter1_ws, "VoterA", role="spectator")
        _join(voter2_ws, "VoterB", role="spectator")

        agent_ws.send_json({
            "type": "candidate_report", "label": "backpack", "confidence": 0.7,
            "instant_confidence": 0.7, "waypoint": "W2", "is_registered_target": False,
        })

        candidate_id, site_id = None, None
        for ws in (agent_ws, voter1_ws, voter2_ws):
            while True:
                msg = ws.receive_json()
                if msg.get("type") == "candidate":
                    candidate_id, site_id = msg["id"], msg["site_id"]
                    break

        voter1_ws.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "dispute"})
        voter2_ws.send_json({"type": "vote", "site_id": site_id, "id": candidate_id, "vote": "dispute"})

        for _ in range(8):
            msg = voter2_ws.receive_json()
            if msg.get("type") == "match_rejected":
                break

        assert "VoterA" not in hub.leaderboard
        assert "VoterB" not in hub.leaderboard
        assert hub.sites[site_id].score == 0


# ---------------------------------------------------------------------------
# CollisionGuard debounce / re-arm state machine -- the actual
# safety-critical logic, not just "does it detect a close object once."
# ---------------------------------------------------------------------------

class _CountingDetector:
    """Fake detector that returns a scripted sequence of bbox-area
    fractions, one per call, then keeps returning the last one. Lets a
    test drive CollisionGuard through a precise sequence of near/far
    reads without needing a real camera or asyncio timing."""

    def __init__(self, area_sequence):
        self.available = True
        self._sequence = list(area_sequence)
        self._calls = 0

    def detect(self, frame):
        idx = min(self._calls, len(self._sequence) - 1)
        self._calls += 1
        area = self._sequence[idx]
        if area <= 0:
            return []
        # a single square-ish bbox whose area matches the scripted fraction
        side = area ** 0.5
        return [FakeDetection((0.0, 0.0, side, side))]


async def _run_guard_n_ticks(guard: CollisionGuard, n: int):
    """Runs CollisionGuard.run()'s loop body n times without real sleep
    delays, by patching asyncio.sleep to a no-op and stopping after n
    iterations via a counting side effect."""
    original_sleep = asyncio.sleep
    calls = {"n": 0}

    async def fast_sleep(_seconds):
        calls["n"] += 1
        if calls["n"] > n:
            raise asyncio.CancelledError()
        return

    asyncio.sleep = fast_sleep
    try:
        await guard.run()
    except asyncio.CancelledError:
        pass
    finally:
        asyncio.sleep = original_sleep


@pytest.mark.asyncio
async def test_collision_guard_trips_when_object_crosses_threshold():
    """A single close-object read must trip `blocked = True` and fire
    exactly one collision_alert -- not require multiple consecutive
    close reads to trip (unlike re-arming, which does debounce)."""
    events = []
    twin = MagicMock()
    twin.publish_command = MagicMock()
    detector = _CountingDetector([0.05, 0.05, PROXIMITY_AREA_THRESHOLD + 0.1])

    guard = CollisionGuard(twin, detector, on_event=events.append)
    await _run_guard_n_ticks(guard, 3)

    assert guard.blocked is True
    alert_events = [e for e in events if e["type"] == "collision_alert"]
    assert len(alert_events) == 1
    twin.publish_command.assert_called_with("stop", {})


@pytest.mark.asyncio
async def test_collision_guard_requires_debounced_clear_streak_to_rearm():
    """This is the actual edge case worth testing: re-arming after a
    stop requires CLEAR_STREAK_REQUIRED consecutive clear reads, not
    just one. A single flaky clear frame must NOT re-arm the guard --
    this is exactly the design flaw the module docstring says was
    caught live (a clean frame let the robot re-drive into a still-
    present obstacle) and fixed by requiring a streak."""
    events = []
    twin = MagicMock()
    twin.publish_command = MagicMock()

    # trip once, then go clear for (CLEAR_STREAK_REQUIRED - 1) ticks --
    # should NOT re-arm yet
    near = PROXIMITY_AREA_THRESHOLD + 0.1
    clear = 0.01
    sequence = [near] + [clear] * (CLEAR_STREAK_REQUIRED - 1)
    detector = _CountingDetector(sequence)
    guard = CollisionGuard(twin, detector, on_event=events.append)
    await _run_guard_n_ticks(guard, len(sequence))

    assert guard.blocked is True, (
        f"guard re-armed after only {CLEAR_STREAK_REQUIRED - 1} clear reads -- "
        f"debounce requirement was not enforced"
    )
    clear_events = [e for e in events if e["type"] == "collision_clear"]
    assert len(clear_events) == 0


@pytest.mark.asyncio
async def test_collision_guard_rearms_after_full_clear_streak():
    """One more clear tick than the previous test -- now it SHOULD
    re-arm and fire exactly one collision_clear event."""
    events = []
    twin = MagicMock()
    twin.publish_command = MagicMock()

    near = PROXIMITY_AREA_THRESHOLD + 0.1
    clear = 0.01
    sequence = [near] + [clear] * CLEAR_STREAK_REQUIRED
    detector = _CountingDetector(sequence)
    guard = CollisionGuard(twin, detector, on_event=events.append)
    await _run_guard_n_ticks(guard, len(sequence))

    assert guard.blocked is False
    clear_events = [e for e in events if e["type"] == "collision_clear"]
    assert len(clear_events) == 1


@pytest.mark.asyncio
async def test_collision_guard_bad_frame_read_does_not_crash_loop():
    """A twin.get_frame() failure mid-loop must not kill the guard --
    FMEA #3's flagged gap is the lack of an operator alert here, not a
    crash, but a crash would be strictly worse and this locks that in."""
    events = []
    twin = MagicMock()
    twin.get_frame.side_effect = RuntimeError("simulated camera read failure")
    detector = _CountingDetector([0.01, 0.01])

    guard = CollisionGuard(twin, detector, on_event=events.append)
    # Should complete without raising, despite every frame read failing
    await _run_guard_n_ticks(guard, 2)
    assert guard.blocked is False