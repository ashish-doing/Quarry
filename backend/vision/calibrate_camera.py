"""
calibrate_camera.py -- one-time camera intrinsics calibration for QUARRY.

RUN THIS FIRST, before any AprilTag pose work, and you can do it anywhere
-- your current room, not the venue. AprilTag distance/pose estimates
depend on knowing the camera's real focal length (fx, fy) and optical
center (cx, cy), NOT the FOV number from Cyberwave's asset metadata --
that's a spec-sheet value for the lens design, not a measurement of your
specific unit, and it says nothing about lens distortion, which a ~90-150
degree lens has plenty of, especially toward the frame edges.

Standard OpenCV checkerboard calibration. You need a printed checkerboard
pattern -- a 9x6 internal-corners board (i.e. 10x7 squares) is the
default assumed here; if you use a different board, change
CHECKERBOARD_SIZE and SQUARE_SIZE_M below.

Usage:
    1. Print a checkerboard (search "opencv checkerboard 9x6" for a
       ready-made PDF, or generate one), measure your printed square
       size in meters, set SQUARE_SIZE_M.
    2. python calibrate_camera.py --source webcam    # your laptop cam, quick test
       python calibrate_camera.py --source twin      # the actual UGV camera via Cyberwave SDK
    3. Hold the checkerboard in front of the camera at varied angles/
       distances/positions in frame. Press SPACE to capture a sample
       when the board is detected (outlined in the preview window).
       Aim for 15-20 good captures covering different parts of the frame
       -- corners and edges matter more than the center for distortion.
    4. Press 'q' when done. Writes camera_intrinsics.json.

This calibrates the QUARRY UGV's actual camera, not a generic guess --
don't skip this and hand apriltag_detector.py a made-up fx/fy, since a
plausible-looking wrong distance is worse than the tool simply refusing
to run (which is what apriltag_detector.py does without this file).
"""

import argparse
import json
import os

import cv2
import numpy as np

CHECKERBOARD_SIZE = (9, 6)   # internal corners, (columns, rows) -- NOT square count
SQUARE_SIZE_M =  0.024        # 2.4cm squares -- MEASURE YOUR OWN PRINTOUT, don't trust this default
MIN_CAPTURES = 12
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")


def _object_points():
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_M
    return objp


def _frame_source(source: str):
    """Yields HxWx3 BGR frames. 'webcam' uses cv2.VideoCapture(0) for a
    quick sanity-check calibration; 'twin' pulls the REAL camera via the
    same call real_feed.py uses, which is what you actually want the
    final intrinsics calibrated against -- a laptop webcam and the UGV's
    camera are different lenses with different distortion."""
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0)")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()

    elif source == "twin":
        import os as _os
        from dotenv import load_dotenv
        import cyberwave

        load_dotenv()
        if "CYBERWAVE_API_KEY" not in _os.environ:
            raise RuntimeError("CYBERWAVE_API_KEY not set -- add it to backend/.env")
        cyberwave.configure(api_key=_os.environ["CYBERWAVE_API_KEY"])
        env_id = _os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
        t = cyberwave.twin("waveshare/ugv-beast", environment=env_id, name="QUARRY")
        while True:
            frame = t.get_frame("numpy", source="remote_edge")
            if frame is None:
                continue
            # get_frame returns RGB (per real_feed.py's normalize_frame path) -- cv2 wants BGR
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError(f"unknown source '{source}' -- use 'webcam' or 'twin'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "twin"], default="webcam")
    args = parser.parse_args()

    objp = _object_points()
    objpoints, imgpoints = [], []
    frame_size = None

    print(f"Calibrating against source='{args.source}'. SPACE to capture, 'q' to finish.")
    print(f"Checkerboard: {CHECKERBOARD_SIZE[0]}x{CHECKERBOARD_SIZE[1]} internal corners, "
          f"{SQUARE_SIZE_M*1000:.0f}mm squares -- VERIFY this matches your printout.")

    for frame in _frame_source(args.source):
        frame_size = frame.shape[:2][::-1]  # (width, height)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        display = frame.copy()
        if found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
            )
            cv2.drawChessboardCorners(display, CHECKERBOARD_SIZE, corners, found)

        cv2.putText(display, f"captures: {len(objpoints)}/{MIN_CAPTURES}+ | SPACE=capture q=finish",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  captured {len(objpoints)}")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(objpoints) < MIN_CAPTURES:
        print(f"Only {len(objpoints)} captures -- need at least {MIN_CAPTURES} for a "
              f"trustworthy calibration. Re-run and capture more, at varied angles/distances.")
        return

    print("Running calibration...")
    ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])

    result = {
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "frame_width": frame_size[0], "frame_height": frame_size[1],
        "reprojection_error": float(ret),  # lower is better; >1.0 px is suspect, re-calibrate
        "source": args.source,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    print(f"Reprojection error: {ret:.3f} px "
          f"({'OK' if ret < 1.0 else 'HIGH -- consider re-calibrating with more/better captures'})")
    if args.source == "webcam":
        print("\nNOTE: this calibrated your WEBCAM, not the UGV's camera. Re-run with "
              "--source twin against the real robot before trusting AprilTag distances.")


if __name__ == "__main__":
    main()