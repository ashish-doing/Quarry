"""
Manual CV demo -- run this instead of the autonomous mission tonight.
No navigation, no camera_intrinsics.json needed. You manually teleop the
robot up to an object, press SPACE to capture and run the real detection
pipeline (shape/color + OSNet re-id + OCR) on that single frame. Also
serves the live frame over HTTP on port 8098 (matching inner_agent.py's
port) so the frontend's REAL FEED tab has something to show.
"""
import os
import threading
import cv2
from dotenv import load_dotenv
import cyberwave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from backend.vision.shape_color import detect_shape_and_color
from backend.vision.reid import TargetMatcher
from backend.vision.ocr_check import MarkerOCR

FRAME_SERVER_PORT = 8098  # matches inner_agent.py's port -- frontend expects this for "inner" sites
_latest_jpeg = None


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
    print("SPACE = capture + run detection, q = quit")

    while True:
        frame = t.get_frame("numpy", source="remote_edge")
        if frame is None:
            continue

        ok, buf = cv2.imencode(".jpg", frame)
        if ok:
            _latest_jpeg = buf.tobytes()

        cv2.imshow("manual_cv_demo", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            shape, color = detect_shape_and_color(frame)
            label, score = matcher.best_match(frame)
            ocr_result = ocr.read_number(frame)
            print("\n--- Capture ---")
            print(f"Shape/Color: {shape}, {color}")
            print(f"OSNet match: {label} (score={score:.3f})")
            print(f"OCR reading: {ocr_result}")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()