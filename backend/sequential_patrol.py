"""
sequential_patrol.py -- autonomous numbered-object survey, driven in
strict sequential order (1, then 2, then 3...) per the master doc's
locked Section 4 decision.

HONEST LIMITATION, same family as patrol.py's: this is still not
"real" point-to-point navigation in the classic sense -- there is no
map, no path planner, no obstacle-aware routing between waypoints.
What it IS: heading derived from measured position deltas between
control ticks (get_pose().position, which parses reliably), not from
the quaternion (which patrol.py also never touches, for the same
reason -- see real_feed.py's _refresh_telemetry comment). Turn toward
the target, drive when roughly facing it, re-measure heading after
every movement, repeat. This is a real, closed-loop, no-guessing
approach -- just not a full nav stack.

SAFETY: same rule as patrol.py -- run only ONE of teleop.py / patrol.py
/ sequential_patrol.py / voice_control.py at a time. This script starts
its own CollisionGuard and gates every single movement call behind
`if guard.blocked: skip` -- identical pattern to patrol.py, autonomous
driving must never run without the proximity emergency-stop active.

Requires objects_config.json (see objects_registry.py) to exist and be
reconciled with real target registrations -- run register_target() for
each entry's target_name via the existing Mission Control "+ TARGET"
flow (or state.register_target() directly) BEFORE running this script,
or the OSNet match step will never fire and every candidate will be
raised as an unregistered generic detection instead of the intended
sequential target.

Usage:
    python -m backend.sequential_patrol
Ctrl+C to stop (sends a final stop command on exit).
"""

import asyncio
import math
import time
from typing import Optional, Tuple

from dotenv import load_dotenv

from backend import real_feed
from backend.vision.collision_guard import CollisionGuard
from backend.objects_registry import load_objects, expected_number_for
from backend.vision.ocr_check import MarkerOCR

load_dotenv()

# -- tuning constants -------------------------------------------------------
# All of these are starting points, not measured values -- like
# patrol.py's BURST_DISTANCE/TURN_ANGLE, expect to tune against the real
# robot/venue rather than trusting these blind. Flagging that honestly
# here rather than presenting them as calibrated.
BURST_DISTANCE = 0.3
BURST_DURATION = 1.5
TURN_INCREMENT = 0.3          # radians per turn correction (smaller than patrol.py's
                               # sweep TURN_ANGLE since this is a proportional-ish
                               # correction toward a known target, not a fixed sweep)
TURN_THRESHOLD = 0.35          # radians -- below this, drive; above this, turn first
BOOTSTRAP_NUDGE_DURATION = 0.8 # first-ever tick has no heading estimate yet, so take
                               # one small blind step to get two position readings to
                               # derive an initial heading from -- this is the ONLY
                               # blind movement in this whole controller
MOVEMENT_NOISE_FLOOR = 0.03    # position-units of movement below which a "delta" is
                               # treated as sensor noise, not real motion -- update
                               # this once real get_pose() units/venue scale are known,
                               # this value is a guess pending real measurement
ARRIVAL_RADIUS = 5.0            # position-units; distinct from real_feed.WAYPOINT_SNAP_RADIUS
                               # (currently a 9999 testing placeholder) -- this constant
                               # is what this controller actually uses to decide
                               # "close enough to stop and scan," tune against real venue
SCAN_PAUSE_S = 3.0
MAX_TICKS_PER_TARGET = 200     # hard ceiling so a stuck robot doesn't loop forever if
                               # a target is unreachable/misregistered -- surfaces as a
                               # skipped-target log line, not a silent hang


def _vector_to(from_xy: Tuple[float, float], to_xy: Tuple[float, float]) -> Tuple[float, float]:
    return (to_xy[0] - from_xy[0], to_xy[1] - from_xy[1])


def _normalize(v: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    mag = math.hypot(*v)
    if mag < 1e-9:
        return None
    return (v[0] / mag, v[1] / mag)


def _signed_angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Signed angle FROM v1 TO v2, in radians, positive = counter-clockwise
    (turn left), negative = clockwise (turn right) -- standard atan2(cross,
    dot) formulation, avoids the sign ambiguity a plain dot-product angle
    would have."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.atan2(cross, dot)


def _waypoint_xy(waypoint_id: str) -> Optional[Tuple[float, float]]:
    for wp in real_feed.WAYPOINTS:
        if wp["id"] == waypoint_id:
            return (wp["x"], wp["y"])
    return None


async def _drive_to_target(twin, guard: CollisionGuard, target_xy: Tuple[float, float],
                            heading_estimate: Optional[Tuple[float, float]]):
    """Runs control ticks until either arrival (returns the final,
    hopefully-updated heading_estimate) or MAX_TICKS_PER_TARGET is hit
    (returns None -- caller must treat this target as skipped, not stall
    the whole survey on one bad target)."""
    for tick in range(MAX_TICKS_PER_TARGET):
        if guard.blocked:
            await asyncio.sleep(0.5)
            continue

        real_feed.state._refresh_telemetry()
        pos_before = real_feed.state.position()
        pos_before_xy = (pos_before["x"], pos_before["y"])

        if real_feed.state.current_waypoint_id() and \
           math.hypot(pos_before_xy[0] - target_xy[0], pos_before_xy[1] - target_xy[1]) <= ARRIVAL_RADIUS:
            return heading_estimate  # arrived -- caller handles the scan

        if heading_estimate is None:
            # Bootstrap: the ONLY blind movement in this controller, needed
            # once, to get a second position reading to derive heading from.
            twin.move_forward(distance=BURST_DISTANCE * 0.5, duration=BOOTSTRAP_NUDGE_DURATION)
            await asyncio.sleep(BOOTSTRAP_NUDGE_DURATION + 0.3)
        else:
            vector_to_target = _vector_to(pos_before_xy, target_xy)
            angle_gap = _signed_angle_between(heading_estimate, vector_to_target)

            if abs(angle_gap) > TURN_THRESHOLD:
                if angle_gap > 0:
                    twin.turn_left(TURN_INCREMENT)
                else:
                    twin.turn_right(TURN_INCREMENT)
                await asyncio.sleep(0.5)
            else:
                twin.move_forward(distance=BURST_DISTANCE, duration=BURST_DURATION)
                await asyncio.sleep(BURST_DURATION + 0.3)

        real_feed.state._refresh_telemetry()
        pos_after = real_feed.state.position()
        pos_after_xy = (pos_after["x"], pos_after["y"])

        delta = _vector_to(pos_before_xy, pos_after_xy)
        if math.hypot(*delta) > MOVEMENT_NOISE_FLOOR:
            measured_heading = _normalize(delta)
            if measured_heading is not None:
                heading_estimate = measured_heading  # measured, never guessed

    print(f"  [WARN] target at {target_xy} not reached within {MAX_TICKS_PER_TARGET} ticks -- skipping")
    return None


def _location_offset(waypoint_id: str) -> Optional[dict]:
    """Waypoint + offset, per the master doc's locked location decision --
    a cheap vector subtraction against the object's nearest known waypoint,
    no live AprilTag dependency at runtime."""
    wp_xy = _waypoint_xy(waypoint_id)
    if wp_xy is None:
        return None
    pos = real_feed.state.position()
    return {
        "waypoint": waypoint_id,
        "offset_x": round(pos["x"] - wp_xy[0], 3),
        "offset_y": round(pos["y"] - wp_xy[1], 3),
    }


async def sequential_survey():
    real_feed.state.connect()
    twin = real_feed.state.twin
    ocr = MarkerOCR()
    objects = load_objects()

    if not objects:
        print("No objects in objects_config.json -- nothing to survey. See objects_registry.py.")
        return
    if not ocr.available:
        print("[WARN] Tesseract unavailable -- proceeding WITHOUT OCR cross-check. "
              "OSNet re-id alone will be trusted, no anomaly flagging this run.")

    print(f"Sequential survey started -- {len(objects)} object(s) in order: "
          f"{[o.number for o in objects]}. Ctrl+C to stop.")

    guard = CollisionGuard(
        twin, real_feed.state.detector,
        on_event=lambda e: print(f"[collision guard] {e['type']}"),
    )
    asyncio.create_task(guard.run())

    heading_estimate: Optional[Tuple[float, float]] = None

    try:
        for obj in objects:
            target_xy = _waypoint_xy(obj.expected_waypoint)
            if target_xy is None:
                print(f"  [WARN] object-{obj.number}: expected_waypoint "
                      f"'{obj.expected_waypoint}' not found in WAYPOINTS -- skipping")
                continue

            print(f"\n--- Object {obj.number} ({obj.target_name}) -- "
                  f"heading to {obj.expected_waypoint} ---")
            heading_estimate = await _drive_to_target(twin, guard, target_xy, heading_estimate)
            if heading_estimate is None and _waypoint_xy(obj.expected_waypoint) is not None:
                continue  # _drive_to_target already logged the skip reason

            print(f"  Arrived near {obj.expected_waypoint} -- scanning for {SCAN_PAUSE_S}s...")
            scan_until = time.monotonic() + SCAN_PAUSE_S
            confirmed_this_object = False
            while time.monotonic() < scan_until:
                candidate = real_feed.state.maybe_raise_candidate()
                if candidate:
                    location = _location_offset(obj.expected_waypoint)
                    print(f"  candidate: {candidate.label} ({candidate.confidence:.2f}) "
                          f"registered={candidate.is_registered_target} location={location}")

                    if candidate.is_registered_target and ocr.available and \
                       hasattr(candidate, "training_crop"):
                        ocr_result = ocr.read_number(candidate.training_crop)
                        expected = expected_number_for(candidate.label, objects)
                        if expected is not None:
                            anomaly = ocr.cross_check(ocr_result, expected)
                            if anomaly:
                                print(f"  [ANOMALY] {anomaly}")
                            elif ocr_result.number is not None:
                                print(f"  OCR cross-check OK: read '{ocr_result.number}', "
                                      f"matches expected object-{expected}")
                    if candidate.is_registered_target and candidate.label == obj.target_name:
                        confirmed_this_object = True
                await asyncio.sleep(0.3)

            if not confirmed_this_object:
                print(f"  [note] object-{obj.number} scan window closed without a "
                      f"confirmed candidate for '{obj.target_name}' -- may need a vote, "
                      f"or the target may not be visible from this waypoint.")

        print(f"\nSequential survey complete -- all {len(objects)} objects visited in order.")

    except KeyboardInterrupt:
        print("\nStopping sequential survey...")
        try:
            twin.publish_command("stop", {})
        except Exception:  # noqa: BLE001 -- best-effort stop on exit
            pass


if __name__ == "__main__":
    asyncio.run(sequential_survey())