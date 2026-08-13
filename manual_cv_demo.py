"""
Manual CV demo -- run this instead of the autonomous mission tonight.
No navigation, no camera_intrinsics.json needed. You manually teleop the
robot (via Cyberwave's own teleop controls) up to an object, optionally
tag which waypoint you're at (keys 1-8), then press SPACE to capture and
run the real detection pipeline (shape/color + OSNet re-id + OCR) on that
single frame.

Two things now happen on every SPACE capture:
  1. The real detection results print to this terminal, as before.
  2. NEW: this script also joins the Hub as the real Inner Field Agent
     and pushes the capture -- including the actual JPEG it just took --
     as a live candidate_report. That's what makes it show up in
     Active Sightings on the frontend, live, for Spectators to
     CONFIRM/DISPUTE in real time on camera.

It also still serves the live frame over HTTP on port 8098 (matching the
"inner" port the frontend expects) so the REAL FEED tab has something to
show even between captures.

If the Hub isn't reachable (e.g. you haven't started
`uvicorn backend.main:app --port 8001` yet), this degrades gracefully --
detection still runs and prints locally, it just won't appear in the UI.

The Hub client auto-reconnects (retry loop below) instead of connecting
once -- the earlier one-shot version would silently and permanently drop
out of the Hub on the first disconnect (observed happening ~11s after
join, most likely a ping/pong timeout caused by the blocking cv2 loop
holding the GIL long enough that the background asyncio thread can't
service the websocket in time). This is a band-aid, not a fix for that
underlying timeout -- but it means Inner comes back on its own instead
of requiring you to notice and restart the script mid-demo.
"""
import os
import base64
import threading
import asyncio
import json
import cv2
from dotenv import load_dotenv
import cyberwave
import websockets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.vision.shape_color import detect_shape_and_color
from backend.vision.reid import TargetMatcher
from backend.vision.ocr_check import MarkerOCR

FRAME_SERVER_PORT = 8098  # matches inner_agent.py's port -- frontend expects this for "inner" sites
_latest_jpeg = None

# --- Hub connection (this script now IS the live Inner Field Agent) --------
HUB_WS = os.environ.get("QUARRY_HUB_WS", "ws://127.0.0.1:8001/ws")
AGENT_NAME = os.environ.get("QUARRY_AGENT_NAME", "Inner")
AGENT_LOCATION = os.environ.get("QUARRY_LOCATION", "Room A")
HUB_RECONNECT_DELAY_S = 3

_ws_loop = None
_ws_conn = None
_ws_ready = threading.Event()


def _start_frame_server():
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/latest.jpg") and _latest_jpeg is not None:
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(_latest_jpeg)
            else:
                self.send_response(503 if self.path.startswith("/latest.jpg") else 404)
                self.end_headers()

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("0.0.0.0", FRAME_SERVER_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Serving live frame: http://127.0.0.1:{FRAME_SERVER_PORT}/latest.jpg")


def _start_hub_client():
    """Runs a websocket client to the Hub on its own event loop in a
    background thread, so the blocking cv2 loop in main() doesn't need to
    become async. Joins as the real Inner Field Agent, and keeps
    reconnecting if the connection drops (see module docstring)."""

    def runner():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        _ws_loop.run_until_complete(_hub_client_main())

    threading.Thread(target=runner, daemon=True).start()


async def _hub_client_main():
    global _ws_conn
    while True:
        try:
            async with websockets.connect(HUB_WS) as ws:
                _ws_conn = ws
                await ws.send(json.dumps({
                    "type": "join", "name": AGENT_NAME, "role": "field_agent",
                    "location": AGENT_LOCATION, "site_role": "inner", "avatar": "\u25c8",
                }))
                reply = json.loads(await ws.recv())
                if reply.get("type") == "join_rejected":
                    print(f"[Hub] join rejected: {reply.get('reason')} "
                          f"-- is another Inner site already connected (e.g. simulate_squad.py --mode both)?")
                    _ws_ready.set()
                    return  # rejection is a config problem, not a transient drop -- don't loop forever on it
                print(f"[Hub] joined as Inner Field Agent '{AGENT_NAME}' -- "
                      f"SPACE captures will now appear live in Active Sightings.")
                _ws_ready.set()
                async for _ in ws:  # drain incoming (votes, chat, etc.) so the socket doesn't back up
                    pass
        except Exception as e:
            print(f"[Hub] connection lost/failed ({e}) -- retrying in {HUB_RECONNECT_DELAY_S}s...")
        _ws_conn = None
        _ws_ready.set()
        await asyncio.sleep(HUB_RECONNECT_DELAY_S)


def _send_candidate_report(payload):
    """Thread-safe: schedule a send from the cv2 thread onto the hub client's loop."""
    if _ws_conn is None or _ws_loop is None:
        return
    async def _send():
        try:
            await _ws_conn.send(json.dumps(payload))
        except Exception as e:
            print(f"[Hub] send failed: {e}")
    asyncio.run_coroutine_threadsafe(_send(), _ws_loop)


def main():
    global _latest_jpeg

    load_dotenv()
    if "CYBERWAVE_API_KEY" not in os.environ:
        raise RuntimeError("CYBERWAVE_API_KEY not set -- add it to backend/.env")
    cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
    t = cyberwave.twin("waveshare/ugv-beast", environment=env_id, name="QUARRY")

    print("Loading OSNet gallery and OCR engine...")
    matcher = TargetMatcher()
    matcher.load_gallery()
    ocr = MarkerOCR()
    print(f"OCR available: {ocr.available}")  # property, not a method call

    _start_frame_server()
    _start_hub_client()
    _ws_ready.wait(timeout=5)  # give the join a moment before you start capturing

    current_waypoint = "W1"
    print("\nControls: 1-8 = set current waypoint (do this before SPACE), "
          "SPACE = capture + run detection + send to Hub, q = quit\n")

    while True:
        frame = t.get_frame("numpy", source="remote_edge")
        if frame is None:
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            _latest_jpeg = buf.tobytes()

        display = frame.copy()
        cv2.putText(display, f"waypoint: {current_waypoint}  (1-8 to change, SPACE to capture)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("manual_cv_demo", display)
        key = cv2.waitKey(1) & 0xFF

        if ord("1") <= key <= ord("8"):
            current_waypoint = f"W{chr(key)}"
            print(f"[Waypoint] set to {current_waypoint}")

        elif key == ord(" "):
            shape, color = detect_shape_and_color(frame)
            label, score = matcher.best_match(frame)
            ocr_result = ocr.read_number(frame)
            print("\n--- Capture ---")
            print(f"Waypoint: {current_waypoint}")
            print(f"Shape/Color: {shape}, {color}")
            print(f"OSNet match: {label} (score={score:.3f})")
            print(f"OCR reading: {ocr_result}")

            ok2, buf2 = cv2.imencode(".jpg", frame)
            image_b64 = base64.b64encode(buf2.tobytes()).decode("ascii") if ok2 else None

            _send_candidate_report({
                "type": "candidate_report",
                "label": label or "unknown",
                "confidence": float(score) if score is not None else 0.0,
                "instant_confidence": float(score) if score is not None else 0.0,
                "waypoint": current_waypoint,
                "is_registered_target": bool(label),
                "ocr_number": ocr_result,
                "ocr_anomaly": None,
                "detected_shape": shape,
                "detected_color": color,
                "twin_semantic_label": None,
                "image": image_b64,
            })
            if _ws_conn is not None:
                print("[Hub] sent -- check Active Sightings in the frontend, Spectators can vote now.")
            else:
                print("[Hub] not connected -- capture only printed locally, nothing sent.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()