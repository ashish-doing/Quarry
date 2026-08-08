"""
site_agent.py -- Field Agent relay, run on YOUR laptop alongside your own
robot. Connects to the shared Hub (main.py, possibly running on someone
else's machine via ngrok) as a WebSocket client and reports YOUR site's
telemetry + candidates into the shared session.

This does NOT give the Hub or any spectator control over your robot --
it only sends data outward (site_report, candidate_report). Nothing in
this script or the Hub's contract accepts a drive command from anyone
but your own local teleop.py / real_feed.py, which you still run exactly
as before, unchanged.

Reuses real_feed.py's RealRobotState so detection/pose logic lives in
exactly one place -- this file only adds the "relay it to the Hub" concern.

Usage:
    python backend/site_agent.py --hub ws://<hub-host>:8000/ws --name "Priya" --location "Berlin, DE"

Requires: pip install websockets
"""

import argparse
import asyncio
import cgi
import json
import os
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import websockets
from dotenv import load_dotenv

from backend.vision.collision_guard import CollisionGuard

from backend import real_feed

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s")

load_dotenv()

FRAME_SERVER_PORT = 8099
_latest_jpeg = None  # shared between the detection loop and the HTTP thread


# One fixed, high-contrast color per label so different object types are
# instantly distinguishable in the live feed, instead of everything being
# the same green. BGR tuples (OpenCV convention, not RGB). Cycles by hash
# if a label isn't in this explicit map, so new vocabulary entries still
# get a stable, distinct-ish color without needing a manual update here.
_LABEL_COLORS = {
    "person":    (0, 0, 255),     # red -- highest-attention label
    "clothing":  (128, 128, 128), # gray -- deliberately dull, the "false alarm" label
    "backpack":  (0, 255, 255),   # yellow
    "bag":       (0, 200, 255),   # orange-yellow
    "box":       (255, 128, 0),   # blue-ish
    "toolbox":   (255, 0, 128),   # purple-ish
    "chair":     (0, 255, 0),     # green
}
_FALLBACK_PALETTE = [
    (255, 0, 0), (0, 128, 255), (255, 0, 255), (128, 255, 0),
    (0, 200, 200), (200, 200, 0), (180, 0, 180),
]


def _color_for_label(label: str):
    if label in _LABEL_COLORS:
        return _LABEL_COLORS[label]
    return _FALLBACK_PALETTE[hash(label) % len(_FALLBACK_PALETTE)]


# ---------------------------------------------------------------------------
# Cube-outline overlay -- per the master doc's locked Section 4 decision:
# "Cube outline -- stylized 2D overlay ONLY, not real depth." This hardware
# has no lidar/depth camera (see collision_guard.py's own honesty note about
# the same hardware ceiling), so there is nothing real to draw a 3D box
# from. What follows is a fixed-fraction visual flourish -- a "back face"
# offset by a constant percentage of the box's own width/height, connected
# to the real 2D bbox corners -- for pitch-material polish only. NEVER
# read this as, or present it as, an actual 3D reconstruction.
# ---------------------------------------------------------------------------
CUBE_DEPTH_FRACTION = 0.18  # how far the "back face" is offset, as a fraction of the
                             # box's own width/height -- a styling constant, not a
                             # measurement of anything


def _draw_cube_overlay(frame, x1: int, y1: int, x2: int, y2: int, color):
    """Draws a pseudo-3D wireframe box over an existing 2D bbox. Purely a
    rendering flourish (see module note above) -- offsets every box by the
    same fixed convention (back face up-and-right) so the effect reads
    consistently across the whole annotated feed, rather than trying to
    infer a "correct" direction from anything the vision pipeline actually
    knows (it doesn't know one -- single RGB frame, no depth)."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    dx, dy = max(2, int(w * CUBE_DEPTH_FRACTION)), max(2, int(h * CUBE_DEPTH_FRACTION))
    bx1, by1, bx2, by2 = x1 + dx, y1 - dy, x2 + dx, y2 - dy

    thin = 1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thin)
    for (fx, fy), (bxp, byp) in (
        ((x1, y1), (bx1, by1)), ((x2, y1), (bx2, by1)),
        ((x2, y2), (bx2, by2)), ((x1, y2), (bx1, by2)),
    ):
        cv2.line(frame, (fx, fy), (bxp, byp), color, thin)


def _draw_detections(frame, detections):
    """Draw a box + label/confidence + pseudo-3D cube overlay per detection,
    one distinct color per label so different object types are easy to tell
    apart at a glance. The cube overlay is 2D-styled polish only -- see
    _draw_cube_overlay's docstring."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        color = _color_for_label(det.label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        _draw_cube_overlay(annotated, x1, y1, x2, y2, color)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "training_data")
PENDING_CROP_MAXLEN = 20  # not-yet-echoed-by-the-Hub crops kept in memory; old
                           # unmatched entries just age out if a broadcast is lost


def _slugify(name: str) -> str:
    """Mirrors main.py's own _slugify exactly -- needed here so this process
    can recognize which Hub broadcasts are about ITS OWN site, without
    importing main.py's FastAPI app just for one string function."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "site"


def _save_training_example(label: str, crop, outcome: str):
    """outcome: 'confirmed' or 'disputed'. This is the data-flywheel itself:
    every vote is a human-verified label. Nothing here trains anything --
    this only accumulates a real, correctly-labeled dataset. Fine-tuning
    OSNet's head on it is a deliberate follow-up, run once there's enough
    real volume to make fine-tuning meaningful, not guessed against zero
    data now."""
    if crop is None or crop.size == 0:
        return
    folder = os.path.join(TRAINING_DATA_DIR, outcome, label)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{int(time.time() * 1000)}.jpg")
    try:
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    except Exception as exc:  # noqa: BLE001 -- a bad save must never crash the relay loop
        print(f"[training data] save failed: {exc}")
        return
    _update_stats(label, outcome)


def _update_stats(label: str, outcome: str):
    """Running confirm/dispute tally per label -- the closest thing this
    project has to an evaluation metric, grounded in real live voting
    instead of an offline benchmark."""
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    stats_path = os.path.join(TRAINING_DATA_DIR, "stats.json")
    stats = {}
    if os.path.exists(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:  # noqa: BLE001 -- a corrupt stats file must not crash the loop
            stats = {}
    entry = stats.setdefault(label, {"confirmed": 0, "disputed": 0})
    entry[outcome] = entry.get(outcome, 0) + 1
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

class _FrameHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/latest.jpg"):
            if _latest_jpeg is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(_latest_jpeg)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith("/register-target"):
            self._handle_register_target()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_register_target(self):
        """Local-only target registration for THIS Field Agent's own
        matcher -- same same-machine-only pattern as /latest.jpg. Photos
        are decoded here and handed to real_feed.state.register_target(),
        which is otherwise never called by anything in the current
        codebase -- registering a name/points on the shared Hub via
        /api/targets does NOT feed this; the two are separate concerns
        by design (shared scoring metadata vs. local CV matching)."""
        try:
            environ = {"REQUEST_METHOD": "POST"}
            if self.headers.get("Content-Type"):
                environ["CONTENT_TYPE"] = self.headers["Content-Type"]
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ=environ)

            name = (form.getvalue("name") or "").strip()
            note = form.getvalue("note") or ""
            if not name:
                self._json_response(400, {"error": "name required"})
                return

            photo_fields = form["photos"] if "photos" in form else []
            if not isinstance(photo_fields, list):
                photo_fields = [photo_fields]

            decoded = []
            for field in photo_fields:
                data = field.file.read() if hasattr(field, "file") and field.file else None
                if not data:
                    continue
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    decoded.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if not decoded:
                self._json_response(400, {"error": "no valid photos received"})
                return

            real_feed.state.register_target(name, note, photos=decoded)
            self._json_response(200, {"status": "registered", "name": name, "photos_used": len(decoded)})
        except Exception as exc:  # noqa: BLE001 -- a bad upload must not kill the frame server
            self._json_response(500, {"error": str(exc)})

    def _json_response(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 -- silence default request logging spam
        pass


def _start_frame_server():
    server = ThreadingHTTPServer(("0.0.0.0", FRAME_SERVER_PORT), _FrameHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Annotated video feed: http://127.0.0.1:{FRAME_SERVER_PORT}/latest.jpg "
          f"(SAME-MACHINE ONLY for now -- a remote spectator can't reach this yet, "
          f"that needs the still-pending ngrok/tunnel work)")


async def run(hub_url: str, name: str, location: str, team: str):
    real_feed.state.connect()
    print(f"Connected to your own twin. Relaying to Hub as Field Agent '{name}' ({location}).")
    _start_frame_server()

    async with websockets.connect(hub_url) as ws:
        await ws.send(json.dumps({
            "type": "join", "name": name, "role": "field_agent", "location": location, "team": team,
        }))

        init_raw = await ws.recv()
        init = json.loads(init_raw)
        if init.get("type") == "join_rejected":
            print(f"Hub rejected join: {init.get('reason')}")
            return
        print("Joined Hub session.")

        # Collision guard was previously only started inside real_feed.py's
        # telemetry_loop(), which the multi-site path never calls -- the
        # emergency stop has never actually run during a real site_agent.py
        # session. Started here instead, now that a ws connection exists
        # to report alerts through.
        my_site_id = _slugify(name)
        pending_crops = deque(maxlen=PENDING_CROP_MAXLEN)
        hub_id_to_crop = {}  # Hub-assigned candidate id -> {"label", "crop"}

        def _on_collision_event(event: dict):
            asyncio.create_task(ws.send(json.dumps({
                "type": "safety_event", "event": event["type"], "timestamp": event["timestamp"],
            })))

        guard = CollisionGuard(real_feed.state.twin, real_feed.state.detector, on_event=_on_collision_event)
        asyncio.create_task(guard.run())

        # Now actively processes broadcasts (previously dropped everything) --
        # correlates the Hub's own candidate id back to the crop THIS process
        # raised, purely by matching (site_id, label, waypoint) against the
        # pending queue. The Hub itself never sees or stores crop images.
        async def drain():
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:  # noqa: BLE001
                    continue

                if msg.get("type") == "candidate" and msg.get("site_id") == my_site_id:
                    for i, pending in enumerate(pending_crops):
                        if pending["label"] == msg.get("label") and pending["waypoint"] == msg.get("waypoint"):
                            hub_id_to_crop[msg["id"]] = pending
                            del pending_crops[i]
                            break

                elif msg.get("type") == "match_confirmed" and msg.get("id") in hub_id_to_crop:
                    entry = hub_id_to_crop.pop(msg["id"])
                    _save_training_example(entry["label"], entry["crop"], "confirmed")

                elif msg.get("type") == "match_rejected" and msg.get("id") in hub_id_to_crop:
                    entry = hub_id_to_crop.pop(msg["id"])
                    _save_training_example(entry["label"], entry["crop"], "disputed")

        asyncio.create_task(drain())

        while True:
            real_feed.state._refresh_telemetry()
            await ws.send(json.dumps({
                "type": "site_report",
                "waypoints": real_feed.WAYPOINTS,
                "telemetry": {
                    "position": real_feed.state.position(),
                    "heading": real_feed.state.heading(),
                    "speed": real_feed.state.speed(),
                    "waypoint": real_feed.state.current_waypoint_id(),
                    "battery": real_feed.state._battery,
                    "fps": real_feed.state._fps,
                },
            }))

            # -- video overlay: runs every tick, independent of the
            # waypoint/cooldown gating that maybe_raise_candidate() uses
            # for scoring -- you should see boxes regardless of whether
            # a "candidate" is currently being raised for points.
            # One capture + one detect() per tick, shared between the video
            # overlay and candidate scoring below -- previously ran twice.
            frame = real_feed.state._capture_frame()
            detections = []
            if frame is not None and real_feed.state.detector.available:
                frame = real_feed.state.detector.normalize_frame(frame)
                if frame is not None:
                    detections = real_feed.state.detector.detect(frame)
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    annotated = _draw_detections(bgr, detections)
                    ok, buf = cv2.imencode(".jpg", annotated)
                    if ok:
                        global _latest_jpeg
                        _latest_jpeg = buf.tobytes()

            candidate = real_feed.state.maybe_raise_candidate(frame=frame, detections=detections)
            if candidate:
                crop = getattr(candidate, "training_crop", None)
                if crop is not None:
                    pending_crops.append({
                        "label": candidate.label, "waypoint": candidate.waypoint, "crop": crop,
                    })
                await ws.send(json.dumps({
                    "type": "candidate_report",
                    "label": candidate.label,
                    "confidence": candidate.confidence,
                    "instant_confidence": candidate.instant_confidence,
                    "waypoint": candidate.waypoint,
                    "is_registered_target": candidate.is_registered_target,
                    # Additive: waypoint+offset location + OCR cross-check,
                    # computed in real_feed.maybe_raise_candidate() -- just
                    # relayed here, not recomputed.
                    "location_offset": getattr(candidate, "location_offset", None),
                    "ocr_number": getattr(candidate, "ocr_number", None),
                    "ocr_anomaly": getattr(candidate, "ocr_anomaly", None),
                }))

            await asyncio.sleep(real_feed.TELEMETRY_TICK_S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, help="ws:// or wss:// URL of the Hub's /ws endpoint")
    parser.add_argument("--name", required=True)
    parser.add_argument("--location", required=True, help='e.g. "Berlin, DE" -- shown to everyone, keep it coarse')
    parser.add_argument("--team", default="Alpha", choices=["Alpha", "Bravo"])
    args = parser.parse_args()

    asyncio.run(run(args.hub, args.name, args.location, args.team))