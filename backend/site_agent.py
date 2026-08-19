"""
site_agent.py -- OUTER UGV relay, run on an outer participant's own
laptop alongside their own robot. Connects to the shared Hub (main.py)
as a WebSocket client and reports that site's telemetry, candidates,
and intruder alerts into the shared mission session.

MISSION ROLE: outer UGVs patrol their own space as overwatch -- they
don't run the numbered-object sequence (that's the inner UGV's job, see
inner_agent.py). What an outer UGV DOES do: share live position with
everyone else (so the inner UGV's operator can see roughly where outer
coverage is), and raise an immediate alert if its own vision spots an
unregistered person -- a possible intruder near the mission. Movement
itself is still driven separately (teleop.py or patrol.py, exactly as
before) -- this script only relays and watches, never drives.

This does NOT give the Hub or any spectator control over your robot --
it only sends data outward. Nothing in this script or the Hub's
contract accepts a drive command from anyone but your own local
teleop.py / patrol.py, which you still run exactly as before, unchanged.

Reuses real_feed.py's RealRobotState so detection/pose logic lives in
exactly one place -- this file only adds the "relay it to the Hub"
concern, same as it always did.

Usage:
    python -m backend.site_agent --hub ws://<hub-host>:8000/ws --name "Priya" --location "Berlin, DE"

Requires: pip install websockets
"""

import argparse
import asyncio
import base64
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
from backend.vision.overlay import draw_detections
from backend.objects_registry import twin_label_for

from backend import real_feed

import logging
logging.basicConfig(level=logging.DEBUG, format="%(levelname)s [%(name)s] %(message)s")

load_dotenv()

FRAME_SERVER_PORT = 8099
_latest_jpeg = None  # shared between the detection loop and the HTTP thread

# -- live WS frame relay -- same mechanism as inner_agent.py, see that
# file's comment for the full reasoning. This is what makes REAL FEED work
# for a spectator who isn't on the same machine as this outer UGV.
FRAME_RELAY_INTERVAL_S = 0.2
FRAME_RELAY_MAX_DIM = 640
_last_frame_relay_at = 0.0


def _encode_relay_frame(bgr) -> "str | None":
    h, w = bgr.shape[:2]
    scale = min(1.0, FRAME_RELAY_MAX_DIM / max(h, w))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


async def _maybe_relay_frame(ws, bgr):
    global _last_frame_relay_at
    now = time.monotonic()
    if now - _last_frame_relay_at < FRAME_RELAY_INTERVAL_S:
        return
    _last_frame_relay_at = now
    encoded = _encode_relay_frame(bgr)
    if encoded is None:
        return
    try:
        await ws.send(json.dumps({"type": "frame", "image": encoded}))
    except Exception as exc:  # noqa: BLE001 -- a dropped frame must never kill the relay loop
        print(f"[Hub] frame relay send failed ({exc}) -- will retry next tick")

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "training_data")
PENDING_CROP_MAXLEN = 20  # not-yet-echoed-by-the-Hub crops kept in memory; old
                           # unmatched entries just age out if a broadcast is lost
PROOF_IMAGE_MAX_DIM = 480  # downscale crops before base64-encoding for the Hub --
                            # this is a proof-gallery thumbnail, not a full-res photo,
                            # keep the WebSocket payload small


def _slugify(name: str) -> str:
    """Mirrors main.py's own _slugify exactly -- needed here so this process
    can recognize which Hub broadcasts are about ITS OWN site, without
    importing main.py's FastAPI app just for one string function."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "site"


def _encode_proof_image(crop) -> str:
    """Downscales and JPEG-encodes a crop for transmission to the Hub --
    this is the mission's proof-gallery image, sent once per confirmed-
    candidate-worthy capture, not per frame. Reverses the earlier
    same-machine-only crop rule specifically for this purpose (see
    main.py's module docstring note on `image`)."""
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    scale = min(1.0, PROOF_IMAGE_MAX_DIM / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


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
        """Local-only, OFFLINE pre-run target registration -- there is no
        live in-session '+TARGET' UI anymore (see main.py's docstring on
        what was removed). This endpoint still exists as a same-machine
        scripting tool: run it once before a session starts to register
        this outer site's own targets, if it has any. For the inner UGV's
        numbered-object registration, see register_mission_objects.py /
        inner_agent.py instead -- outer sites typically won't call this
        at all, since outer UGVs don't run the numbered-object sequence."""
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


async def run(hub_url: str, name: str, location: str, site_role: str, code: str = ""):
    real_feed.state.connect()
    print(f"Connected to your own twin. Relaying to Hub as {site_role.title()} "
          f"Field Agent '{name}' ({location}).")
    _start_frame_server()

    async with websockets.connect(hub_url) as ws:
        await ws.send(json.dumps({
            "type": "join", "name": name, "role": "field_agent",
            "location": location, "site_role": site_role, "code": code,
        }))

        init_raw = await ws.recv()
        init = json.loads(init_raw)
        if init.get("type") == "join_rejected":
            print(f"Hub rejected join: {init.get('reason')}")
            return
        print("Joined Hub session.")

        my_site_id = _slugify(name)
        pending_crops = deque(maxlen=PENDING_CROP_MAXLEN)
        hub_id_to_crop = {}  # Hub-assigned candidate id -> {"label", "crop"}

        def _on_collision_event(event: dict):
            asyncio.create_task(ws.send(json.dumps({
                "type": "safety_event", "event": event["type"], "timestamp": event["timestamp"],
            })))

        guard = CollisionGuard(real_feed.state.twin, real_feed.state.detector, on_event=_on_collision_event)
        asyncio.create_task(guard.run())

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
            # waypoint/cooldown gating that maybe_raise_candidate() uses.
            # One capture + one detect() per tick, shared between the video
            # overlay, candidate scoring, and intruder-check below.
            frame = real_feed.state._capture_frame()
            detections = []
            if frame is not None and real_feed.state.detector.available:
                frame = real_feed.state.detector.normalize_frame(frame)
                if frame is not None:
                    detections = real_feed.state.detector.detect(frame)
                    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    annotated = draw_detections(bgr, detections)
                    ok, buf = cv2.imencode(".jpg", annotated)
                    if ok:
                        global _latest_jpeg
                        _latest_jpeg = buf.tobytes()
                    await _maybe_relay_frame(ws, annotated)

            # -- intruder watch: this is the core of the outer-UGV role.
            # Independent of candidate-raising below -- fires its own
            # cooldown, no waypoint gating (see real_feed.check_intruder's
            # docstring for why).
            alert = real_feed.state.check_intruder(detections)
            if alert:
                await ws.send(json.dumps({"type": "alert_report", **alert}))
                print(f"[ALERT] {alert['message']} near {alert['nearest_waypoint']}")

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
                    "location_offset": getattr(candidate, "location_offset", None),
                    "ocr_number": getattr(candidate, "ocr_number", None),
                    "ocr_anomaly": getattr(candidate, "ocr_anomaly", None),
                    "image": _encode_proof_image(crop),
                    "twin_semantic_label": twin_label_for(candidate.label, real_feed.state.objects_registry),
                }))

            await asyncio.sleep(real_feed.TELEMETRY_TICK_S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, help="ws:// or wss:// URL of the Hub's /ws endpoint")
    parser.add_argument("--name", required=True)
    parser.add_argument("--location", required=True, help='e.g. "Berlin, DE" -- shown to everyone, keep it coarse')
    parser.add_argument("--role", default="outer", choices=["inner", "outer"],
                         help="Mission role for this site. Outer UGVs use this script; the inner "
                              "UGV should normally run inner_agent.py instead, which also drives the "
                              "sequential survey. --role inner is accepted here only for flexibility "
                              "(e.g. relaying an inner UGV that's being driven by a separate teleop.py "
                              "session rather than inner_agent.py's own driving loop).")
    parser.add_argument("--code", default="", help="Session join code, only needed if the Hub "
                                                     "has QUARRY_JOIN_CODE set.")
    args = parser.parse_args()

    asyncio.run(run(args.hub, args.name, args.location, args.role, code=args.code))