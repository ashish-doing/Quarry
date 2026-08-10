"""
inner_agent.py -- the INNER UGV's mission process: drives the sequential
numbered-object survey (same locked Section 4 algorithm as the earlier
sequential_patrol.py) AND relays telemetry/candidates/alerts to the
Hub, in one process.

WHY MERGED (unlike outer UGVs, which run site_agent.py for relay and a
separate patrol.py/teleop.py for driving): the inner UGV's mission is
tightly synchronized -- drive to a waypoint, stop, capture a properly-
framed photo, cross-check identity, THEN move to the next object. That
sequencing has to live in one control loop. Splitting driving and
relay into two processes here would mean either duplicating the drive
state in both, or racing two processes over the same frame captures.
Outer UGVs don't have this problem -- they're just patrolling and
watching, so relay (site_agent.py) and driving (patrol.py/teleop.py)
staying separate is fine there.

HONEST LIMITATION, same family as patrol.py's and the earlier
sequential_patrol.py's: this is not "real" point-to-point navigation.
Heading is derived from measured position deltas between control ticks,
never from get_pose()'s quaternion (left deliberately unconverted --
see real_feed.py's _refresh_telemetry comment).

SAFETY: run ONLY this script for driving the inner UGV -- do not also
run teleop.py/patrol.py against the same robot at the same time. This
script starts its own CollisionGuard and gates every movement call
behind `if guard.blocked: skip`, identical to patrol.py's pattern.

Requires objects_config.json (see objects_registry.py) to exist and be
reconciled with real target registrations -- run
register_mission_objects.py (or state.register_target() directly) for
each entry's target_name BEFORE running this script, or the OSNet match
step never fires and every candidate raises as an unregistered generic
detection instead of the intended sequential target.

Usage:
    python -m backend.inner_agent --hub ws://<hub-host>:8000/ws --name "Inner" --location "Room A"
Ctrl+C to stop (sends a final stop command on exit).
"""

import argparse
import asyncio
import base64
import json
import math
import re
import time
from typing import Optional, Tuple

import cv2
import websockets
from dotenv import load_dotenv

from backend import real_feed
from backend.vision.collision_guard import CollisionGuard
from backend.vision.overlay import draw_detections
from backend.vision.shape_color import detect_shape_and_color
from backend.objects_registry import load_objects, expected_number_for, twin_label_for
from backend.nav_math import vector_to, normalize, signed_angle_between, waypoint_xy, location_offset

load_dotenv()

# -- tuning constants -------------------------------------------------------
# Starting points, not measured values -- tune against the real robot/venue.
BURST_DISTANCE = 0.3
BURST_DURATION = 1.5
TURN_INCREMENT = 0.3
TURN_THRESHOLD = 0.35
BOOTSTRAP_NUDGE_DURATION = 0.8
MOVEMENT_NOISE_FLOOR = 0.03
ARRIVAL_RADIUS = 5.0
SCAN_PAUSE_S = 3.0
MAX_TICKS_PER_TARGET = 200

FRAME_SERVER_PORT = 8098  # different port than site_agent.py's 8099, in case
                            # both an inner and an outer agent run on one machine
_latest_jpeg = None

PROOF_IMAGE_MAX_DIM = 480


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "site"


def _encode_proof_image(crop) -> Optional[str]:
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


def _start_frame_server():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading

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

        def log_message(self, format, *args):  # noqa: A002
            pass

    server = ThreadingHTTPServer(("0.0.0.0", FRAME_SERVER_PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Inner UGV annotated feed: http://127.0.0.1:{FRAME_SERVER_PORT}/latest.jpg (same-machine only)")


async def _send_site_report(ws):
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


async def _capture_and_annotate() -> Tuple[Optional[object], list]:
    """One shared capture+detect for driving ticks and scan ticks alike --
    also updates the local annotated-feed JPEG. Returns (frame, detections)."""
    global _latest_jpeg
    frame = real_feed.state._capture_frame()
    if frame is None or not real_feed.state.detector.available:
        return frame, []
    frame = real_feed.state.detector.normalize_frame(frame)
    if frame is None:
        return None, []
    detections = real_feed.state.detector.detect(frame)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    annotated = draw_detections(bgr, detections)
    ok, buf = cv2.imencode(".jpg", annotated)
    if ok:
        _latest_jpeg = buf.tobytes()
    return frame, detections


async def _drive_to_target(ws, twin, guard: CollisionGuard, target_xy: Tuple[float, float],
                            heading_estimate: Optional[Tuple[float, float]]):
    for _tick in range(MAX_TICKS_PER_TARGET):
        await _send_site_report(ws)

        if guard.blocked:
            await asyncio.sleep(0.5)
            continue

        real_feed.state._refresh_telemetry()
        pos_before = real_feed.state.position()
        pos_before_xy = (pos_before["x"], pos_before["y"])

        if real_feed.state.current_waypoint_id() and \
           math.hypot(pos_before_xy[0] - target_xy[0], pos_before_xy[1] - target_xy[1]) <= ARRIVAL_RADIUS:
            return heading_estimate

        if heading_estimate is None:
            twin.move_forward(distance=BURST_DISTANCE * 0.5, duration=BOOTSTRAP_NUDGE_DURATION)
            await asyncio.sleep(BOOTSTRAP_NUDGE_DURATION + 0.3)
        else:
            target_vector = vector_to(pos_before_xy, target_xy)
            angle_gap = signed_angle_between(heading_estimate, target_vector)
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
        delta = vector_to(pos_before_xy, (pos_after["x"], pos_after["y"]))
        if math.hypot(*delta) > MOVEMENT_NOISE_FLOOR:
            measured_heading = normalize(delta)
            if measured_heading is not None:
                heading_estimate = measured_heading

    print(f"  [WARN] target at {target_xy} not reached within {MAX_TICKS_PER_TARGET} ticks -- skipping")
    return None


async def run(hub_url: str, name: str, location: str):
    real_feed.state.connect()
    twin = real_feed.state.twin
    objects = load_objects()
    _start_frame_server()

    if not objects:
        print("No objects in objects_config.json -- nothing to survey. See objects_registry.py.")
        return
    if not real_feed.state.ocr.available:
        print("[WARN] Tesseract unavailable -- proceeding WITHOUT OCR cross-check.")

    async with websockets.connect(hub_url) as ws:
        await ws.send(json.dumps({
            "type": "join", "name": name, "role": "field_agent",
            "location": location, "site_role": "inner",
        }))
        init_raw = await ws.recv()
        init = json.loads(init_raw)
        if init.get("type") == "join_rejected":
            print(f"Hub rejected join: {init.get('reason')}")
            return
        print(f"Joined Hub session as Inner Field Agent '{name}'.")

        guard = CollisionGuard(
            twin, real_feed.state.detector,
            on_event=lambda e: asyncio.create_task(ws.send(json.dumps({
                "type": "safety_event", "event": e["type"], "timestamp": e["timestamp"],
            }))),
        )
        asyncio.create_task(guard.run())

        print(f"Sequential survey started -- {len(objects)} object(s) in order: "
              f"{[o.number for o in objects]}. Ctrl+C to stop.")

        heading_estimate: Optional[Tuple[float, float]] = None

        try:
            for obj in objects:
                target_xy = waypoint_xy(obj.expected_waypoint, real_feed.WAYPOINTS)
                if target_xy is None:
                    print(f"  [WARN] object-{obj.number}: expected_waypoint "
                          f"'{obj.expected_waypoint}' not found -- skipping")
                    continue

                print(f"\n--- Object {obj.number} ({obj.target_name}) -- "
                      f"heading to {obj.expected_waypoint} ---")
                heading_estimate = await _drive_to_target(ws, twin, guard, target_xy, heading_estimate)
                if heading_estimate is None:
                    continue  # _drive_to_target already logged the skip reason

                print(f"  Arrived near {obj.expected_waypoint} -- scanning for {SCAN_PAUSE_S}s...")
                scan_until = time.monotonic() + SCAN_PAUSE_S
                confirmed_this_object = False
                while time.monotonic() < scan_until:
                    frame, detections = await _capture_and_annotate()

                    # Inner UGV can spot an intruder too, same as outer --
                    # checked every scan tick using the frame already captured.
                    alert = real_feed.state.check_intruder(detections)
                    if alert:
                        await ws.send(json.dumps({"type": "alert_report", **alert}))
                        print(f"  [ALERT] {alert['message']} near {alert['nearest_waypoint']}")

                    candidate = real_feed.state.maybe_raise_candidate(frame=frame, detections=detections)
                    if candidate:
                        crop = getattr(candidate, "training_crop", None)
                        loc = location_offset(obj.expected_waypoint, real_feed.state.position(), real_feed.WAYPOINTS)
                        print(f"  candidate: {candidate.label} ({candidate.confidence:.2f}) "
                              f"registered={candidate.is_registered_target} location={loc}")

                        if candidate.is_registered_target and real_feed.state.ocr.available:
                            expected = expected_number_for(candidate.label, objects)
                            if expected is not None and candidate.ocr_anomaly:
                                print(f"  [ANOMALY] {candidate.ocr_anomaly}")
                            elif expected is not None and candidate.ocr_number is not None:
                                print(f"  OCR cross-check OK: read '{candidate.ocr_number}', "
                                      f"matches expected object-{expected}")

                        # Shape + color: secondary identity signal, additive
                        # only. Prefer fields already on the Candidate if
                        # real_feed.py's maybe_raise_candidate() has been
                        # updated to compute them (see shape_color.py) --
                        # fall back to computing here from the same crop so
                        # this still works even if that wiring lands later.
                        detected_shape = getattr(candidate, "detected_shape", None)
                        detected_color = getattr(candidate, "detected_color", None)
                        if detected_shape is None and detected_color is None and crop is not None:
                            detected_shape, detected_color = detect_shape_and_color(crop)
                        if detected_shape or detected_color:
                            print(f"  shape/color: {detected_shape or '?'}/{detected_color or '?'}")

                        await ws.send(json.dumps({
                            "type": "candidate_report",
                            "label": candidate.label,
                            "confidence": candidate.confidence,
                            "instant_confidence": candidate.instant_confidence,
                            "waypoint": candidate.waypoint,
                            "is_registered_target": candidate.is_registered_target,
                            "location_offset": candidate.location_offset,
                            "ocr_number": candidate.ocr_number,
                            "ocr_anomaly": candidate.ocr_anomaly,
                            "detected_shape": detected_shape,
                            "detected_color": detected_color,
                            "image": _encode_proof_image(crop),
                            "twin_semantic_label": twin_label_for(candidate.label, objects),
                        }))

                        if candidate.is_registered_target and candidate.label == obj.target_name:
                            confirmed_this_object = True

                    await _send_site_report(ws)
                    await asyncio.sleep(0.3)

                if not confirmed_this_object:
                    print(f"  [note] object-{obj.number} scan window closed without a "
                          f"confirmed candidate for '{obj.target_name}' -- may need a vote, "
                          f"or the target may not be visible from this waypoint.")

            print(f"\nSequential survey complete -- all {len(objects)} objects visited in order.")
            # Keep relaying telemetry after the survey completes so
            # spectators don't see the inner site go stale/disconnect
            # mid-demo just because the object sequence finished.
            while True:
                await _send_site_report(ws)
                await asyncio.sleep(real_feed.TELEMETRY_TICK_S * 5)

        except KeyboardInterrupt:
            print("\nStopping inner agent...")
            try:
                twin.publish_command("stop", {})
            except Exception:  # noqa: BLE001 -- best-effort stop on exit
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", required=True, help="ws:// or wss:// URL of the Hub's /ws endpoint")
    parser.add_argument("--name", required=True)
    parser.add_argument("--location", required=True, help='e.g. "Room A" -- shown to everyone')
    args = parser.parse_args()

    asyncio.run(run(args.hub, args.name, args.location))