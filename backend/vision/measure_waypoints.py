"""
measure_waypoints.py -- interactive venue tool. Run this AT the demo room,
after calibrate_camera.py (against --source twin) and after the two
reference tags (IDs 6, 7) are taped to two fixed points in the room.

What this actually does: shows live AprilTag detections with distance/
offset from the camera. You physically position the robot at each
waypoint location, tape/place that waypoint's tag where you want it, and
press its number key to record the tag's position AT THAT MOMENT relative
to the two reference tags. This does NOT convert to the exact same (x, y)
frame get_pose() uses internally -- that mapping is a real follow-up
(see real_feed.py's own honesty note on get_pose()'s quaternion). What
this DOES give you: real, physically-verified positions of the 6
waypoints relative to two fixed, known room points, in meters -- enough
to sanity-check whatever get_pose() reports once you're driving, and to
catch a gross sign/scale error before trusting the patrol pattern.

Usage:
    python measure_waypoints.py
Controls:
    0-7  : record the currently-detected tag with that ID
    p    : print current recorded set
    s    : save recorded set to waypoint_measurements.json
    q    : quit
"""

import json
import os

import cv2
from dotenv import load_dotenv

from apriltag_detector import AprilTagDetector, WAYPOINT_TAG_IDS, REFERENCE_TAG_IDS

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "waypoint_measurements.json")


def main():
    load_dotenv()
    detector = AprilTagDetector()
    if not detector.available:
        print("AprilTag detector not available -- check camera_intrinsics.json exists "
              "(run calibrate_camera.py --source twin first) and pupil-apriltags is installed.")
        return

    import cyberwave
    cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
    env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
    t = cyberwave.twin("waveshare/ugv-beast", environment=env_id, name="QUARRY")

    recorded = {}
    print("Live AprilTag view. Position the robot, press the tag's ID key to record it.")
    print(f"Waypoint tags: {WAYPOINT_TAG_IDS} | Reference tags: {REFERENCE_TAG_IDS}")
    print("Record BOTH reference tags (6, 7) FIRST -- everything else is measured relative to them.")

    while True:
        frame = t.get_frame("numpy", source="remote_edge")
        if frame is None:
            continue
        detections = detector.detect(frame)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for d in detections:
            label = detector.label_for(d.tag_id)
            recorded_mark = " [RECORDED]" if d.tag_id in recorded else ""
            print(f"\r  tag {d.tag_id} ({label}): x={d.x_m:+.3f}m y={d.y_m:+.3f}m "
                  f"dist={d.distance_m:.3f}m{recorded_mark}   ", end="", flush=True)
            corner_pts = d.corners.astype(int)
            cv2.polylines(bgr, [corner_pts], True, (0, 255, 0), 2)
            cv2.putText(bgr, f"{label} {d.distance_m:.2f}m", tuple(corner_pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("measure_waypoints", bgr)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            print("\n", json.dumps(recorded, indent=2))
        elif key == ord("s"):
            with open(OUTPUT_PATH, "w") as f:
                json.dump(recorded, f, indent=2)
            print(f"\nSaved {len(recorded)} measurements to {OUTPUT_PATH}")
        elif chr(key).isdigit():
            tag_id = int(chr(key))
            match = next((d for d in detections if d.tag_id == tag_id), None)
            if match is None:
                print(f"\n  tag {tag_id} not currently visible -- point the camera at it first")
                continue
            recorded[tag_id] = {
                "label": detector.label_for(tag_id),
                "x_m": match.x_m, "y_m": match.y_m, "z_m": match.z_m,
                "distance_m": match.distance_m,
            }
            print(f"\n  recorded tag {tag_id} ({detector.label_for(tag_id)})")

    cv2.destroyAllWindows()
    if recorded:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(recorded, f, indent=2)
        print(f"\nFinal save: {len(recorded)} measurements -> {OUTPUT_PATH}")
        print("\nNext step: use the two reference-tag positions (6, 7) to work out the "
              "offset/rotation into whatever frame get_pose() reports, then hand-fill "
              "real_feed.py's WAYPOINTS x/y. This script gives you ground truth to check "
              "that conversion against -- it does not do the conversion for you.")


if __name__ == "__main__":
    main()