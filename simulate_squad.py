"""
Simulates the OUTER field agent (a second player -- you only have one
robot tonight, so this stands in for it) reporting an intruder alert.

The INNER site is now the REAL one: manual_cv_demo.py connects to the
Hub itself and posts live candidate_reports from actual SPACE captures
on your robot's camera. This script used to also simulate an "Inner"
site, but the Hub only allows one Inner site per session ("one per
session" per the role description) -- so by default this script now
only runs the outer_agent, to avoid colliding with the real inner join.

Use --mode both only as a FALLBACK: if manual_cv_demo.py can't reach the
Hub for some reason, this restores the old fully-scripted single-object
sighting so the vote demo still works.

Run this AFTER `uvicorn backend.main:app --port 8001` is already running.
Keep this script running (don't Ctrl+C) so the outer site stays "online"
in the Squad panel while you record.
"""
import argparse
import asyncio
import json
import websockets

HUB = "ws://127.0.0.1:8001/ws"

# --- EDIT THESE THREE LINES to match one of your actually registered objects
#     (only used if you run with --mode inner or --mode both) ---
OBJECT_LABEL = "object-08"      # must match a target_name in objects_config.json
OBJECT_SHAPE_COLOR = ("cone", "pink")
OBJECT_WAYPOINT = "W3"
# ---------------------------------------------------------------------------

async def inner_agent():
    async with websockets.connect(HUB) as ws:
        await ws.send(json.dumps({
            "type": "join", "name": "Inner", "role": "field_agent",
            "location": "Room A", "site_role": "inner", "avatar": "\u25c8",
        }))
        await ws.recv()  # init
        print("[Inner] joined.")

        await asyncio.sleep(2)
        await ws.send(json.dumps({
            "type": "candidate_report",
            "label": OBJECT_LABEL,
            "confidence": 0.91,
            "instant_confidence": 0.88,
            "waypoint": OBJECT_WAYPOINT,
            "is_registered_target": True,
            "ocr_number": OBJECT_LABEL.split("-")[-1],
            "ocr_anomaly": None,
            "detected_shape": OBJECT_SHAPE_COLOR[0],
            "detected_color": OBJECT_SHAPE_COLOR[1],
            "twin_semantic_label": "Ladder",  # deliberate mismatch for the vote demo
        }))
        print(f"[Inner] reported candidate: {OBJECT_LABEL} at {OBJECT_WAYPOINT}")
        print("[Inner] Now open 2 Spectator tabs in the frontend and vote Confirm"
              " on this sighting to push it into the Proof Gallery.")

        # stay connected so the site shows "online" in the Squad panel
        while True:
            await asyncio.sleep(10)


async def outer_agent():
    async with websockets.connect(HUB) as ws:
        await ws.send(json.dumps({
            "type": "join", "name": "Scout", "role": "field_agent",
            "location": "Perimeter", "site_role": "outer", "avatar": "\u25c8",
        }))
        await ws.recv()  # init
        print("[Outer] joined.")

        await asyncio.sleep(6)
        await ws.send(json.dumps({
            "type": "alert_report",
            "kind": "intruder",
            "position": {"x": 1.2, "y": 0.4},
            "nearest_waypoint": "W5",
            "distance_m": 0.8,
            "message": "Unregistered movement near outer perimeter.",
        }))
        print("[Outer] sent intruder alert near W5.")

        while True:
            await asyncio.sleep(10)


async def main(mode):
    tasks = []
    if mode in ("inner", "both"):
        tasks.append(inner_agent())
    if mode in ("outer", "both"):
        tasks.append(outer_agent())
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["inner", "outer", "both"], default="outer",
        help="default 'outer' -- manual_cv_demo.py is now the live Inner site, "
             "so this only runs the scripted Outer/intruder-alert stand-in. "
             "Use 'both' as a fallback if manual_cv_demo.py can't reach the Hub.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.mode))