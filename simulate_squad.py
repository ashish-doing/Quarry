"""
Simulates the INNER field agent (your real UGV's site) reporting one
detection, and an OUTER field agent (a second player -- you only have one
robot tonight, so this stands in for it) reporting an intruder alert.

This does NOT invent votes or confirmations -- those still have to come
from real browser tabs joined as Spectators, clicking Confirm, so the
demo stays honest: the detection is scripted (no second robot), but the
human-in-the-loop verification is real and live.

Run this AFTER `uvicorn backend.main:app --port 8001` is already running.
Keep this script running (don't Ctrl+C) so the two field-agent sites stay
"online" in the Squad panel while you record.
"""
import asyncio
import json
import websockets

HUB = "ws://127.0.0.1:8001/ws"

# --- EDIT THESE THREE LINES to match one of your actually registered objects ---
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


async def main():
    await asyncio.gather(inner_agent(), outer_agent())

if __name__ == "__main__":
    asyncio.run(main())