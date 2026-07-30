"""
patrol.py -- autonomous patrol for QUARRY, laptop-side.

HONEST LIMITATION, read this first: this is NOT true point-to-point
waypoint navigation. get_pose() returns a rotation quaternion, not a
yaw angle, and that conversion was deliberately left unverified in
real_feed.py rather than guessed (see _refresh_telemetry's comment) --
guessing it wrong here would mean the robot turning confidently in the
wrong direction, which is worse than admitting the gap. So this script
doesn't try to "turn to face waypoint 3". What it DOES do, using only
position (x, y), which IS reliably parsed: drive forward in bursts,
check after each burst whether the new position lands within a
waypoint's snap radius, and if so pause to scan there before
continuing, alternating turn direction between bursts to sweep the
space. That's a real autonomous behavior -- just a sweep pattern, not
a navigation stack. If/when the quaternion->yaw math gets verified
against the physical robot, this can be upgraded to real point-to-point
routing without changing anything else here.

SAFETY: run ONE of patrol.py / teleop.py at a time, never both. Only
one process should be sending drive commands to the robot at once --
running both isn't dangerous (the Pi's own 0.5s movement watchdog just
means whichever script isn't currently sending gets overridden) but
it's confusing to operate and not a real use case. This script starts
its own CollisionGuard, independent of whether main.py/real_feed's
telemetry_loop is also running, because autonomous driving must never
run without the proximity emergency-stop active.

Usage:
    python backend/patrol.py
Ctrl+C to stop (sends a final stop command on exit).
"""

import asyncio
import time

from dotenv import load_dotenv

from backend import real_feed
from backend.vision.collision_guard import CollisionGuard

load_dotenv()

BURST_DISTANCE = 0.3
BURST_DURATION = 1.5
SCAN_PAUSE_S = 3.0    # how long to sit and look once a waypoint is reached
TURN_ANGLE = 0.6      # radians per turn between bursts, alternates direction each waypoint


async def patrol_loop():
    real_feed.state.connect()
    twin = real_feed.state.twin
    print("Autonomous patrol started. Ctrl+C to stop.")

    guard = CollisionGuard(
        twin, real_feed.state.detector,
        on_event=lambda e: print(f"[collision guard] {e['type']}"),
    )
    asyncio.create_task(guard.run())

    turn_dir = 1
    last_waypoint = None
    # Coverage tracking -- NOT path planning. Directed steering toward a
    # specific unvisited waypoint needs real heading/yaw, which get_pose()'s
    # quaternion has deliberately never been converted for (see real_feed.py's
    # _refresh_telemetry comment -- guessing it wrong means confidently
    # driving the wrong direction). This is the honest version: know what's
    # been covered, report full-sweep completion, keep sweeping.
    visited = set()
    total_waypoints = len(real_feed.WAYPOINTS)

    try:
        while True:
            if guard.blocked:
                await asyncio.sleep(0.5)
                continue

            twin.move_forward(distance=BURST_DISTANCE, duration=BURST_DURATION)
            await asyncio.sleep(BURST_DURATION + 0.3)

            real_feed.state._refresh_telemetry()
            wp = real_feed.state.current_waypoint_id()

            if wp and wp != last_waypoint:
                last_waypoint = wp
                visited.add(wp)
                print(f"Reached {wp} -- scanning... (coverage: {len(visited)}/{total_waypoints})")
                scan_until = time.monotonic() + SCAN_PAUSE_S
                while time.monotonic() < scan_until:
                    candidate = real_feed.state.maybe_raise_candidate()
                    if candidate:
                        print(f"  candidate: {candidate.label} ({candidate.confidence:.2f})")
                    await asyncio.sleep(0.3)

                if len(visited) >= total_waypoints:
                    print(f"Full sweep complete -- all {total_waypoints} waypoints covered. Resetting coverage.")
                    visited.clear()

                turn_dir *= -1

            if turn_dir > 0:
                twin.turn_left(TURN_ANGLE)
            else:
                twin.turn_right(TURN_ANGLE)
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\nStopping patrol...")
        try:
            twin.publish_command("stop", {})
        except Exception:  # noqa: BLE001 -- best-effort stop on exit
            pass


if __name__ == "__main__":
    asyncio.run(patrol_loop())