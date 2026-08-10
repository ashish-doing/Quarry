"""
measure_waypoints.py -- interactive venue tool. Run this AT the demo
room, after calibrate_camera.py (against --source twin) and after all
8 AprilTags are placed at their intended waypoint locations.

APPROACH (changed from the earlier two-reference-tag design): with 8
objects needing 8 real waypoints, there was no tag budget left for
dedicated anchor tags, so this uses a simpler procedure instead --
DRIVE-AND-CENTER. For each waypoint: physically drive QUARRY (via a
separate teleop.py session running alongside this script) until it is
centered directly on that waypoint's tag -- the same physical act it
will do autonomously later when navigating there for real -- then press
that tag's number key. This script records get_pose()'s (x, y) at that
exact moment as the waypoint's position. The live AprilTag detection
shown on screen is a CONFIRMATION aid only ("yes, I'm centered on tag
3") -- it is not used as a measurement input.

WHY NOT compute each tag's room position from its camera-relative
AprilTag reading instead (avoiding the need to physically drive to
each one)? That fusion requires knowing the robot's heading (yaw) at
the moment of each reading, to rotate the camera-relative offset into
the room frame. This project deliberately never converts get_pose()'s
rotation quaternion into yaw -- see real_feed.py's _refresh_telemetry
comment, "guessed here after getting pose wrong once already." Rather
than reopen that question for this tool, drive-and-center sidesteps it
entirely: it only ever needs get_pose()'s (x, y), never its rotation.

IF A TAG IS PHYSICALLY UNREACHABLE (blocked by furniture, wanted
visible-but-not-approached for the demo layout): there is no automated
fallback for that case in this script. Measure it by hand (tape
measure, relative to a waypoint you did record) and hand-edit that
one entry in the saved JSON before pasting into real_feed.py.

Usage:
    python -m backend.vision.measure_waypoints
    (run teleop.py in a separate terminal/window to actually drive --
    this script only records, it never sends movement commands)
Controls:
    0-7  : record get_pose() as the currently-centered tag's waypoint
    p    : print current recorded set
    s    : save recorded set to waypoint_measurements.json
    q    : quit
"""

import json
import os

import cv2
from dotenv import load_dotenv

from backend.vision.apriltag_detector import AprilTagDetector, WAYPOINT_TAG_IDS
from backend.real_feed import ASSET_KEY, ENVIRONMENT_ID, TWIN_NAME

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "waypoint_measurements.json")


def main():
    load_dotenv()
    detector = AprilTagDetector()
    if not detector.available:
        print("AprilTag detector not available -- check camera_intrinsics.json exists "
              "(run calibrate_camera.py --source twin first) and pupil-apriltags is installed.")
        return

    import cyberwave
    cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
    t = cyberwave.twin(ASSET_KEY, environment=ENVIRONMENT_ID, name=TWIN_NAME)

    recorded = {}
    print("Drive QUARRY (teleop.py, separate window) to center on each tag in turn,")
    print("then press that tag's ID key here to record get_pose() as its waypoint.")
    print(f"Waypoint tags: {WAYPOINT_TAG_IDS}")
    print("No reference tags needed -- get_pose() itself is the shared coordinate frame.\n")

    while True:
        frame = t.get_frame("numpy", source="remote_edge")
        if frame is None:
            continue
        detections = detector.detect(frame)  # on-screen confirmation only, not a measurement input

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for d in detections:
            label = detector.label_for(d.tag_id)
            recorded_mark = " [RECORDED]" if d.tag_id in recorded else ""
            print(f"\r  tag {d.tag_id} ({label}): dist={d.distance_m:.3f}m{recorded_mark}   ",
                  end="", flush=True)
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
            if tag_id not in WAYPOINT_TAG_IDS:
                print(f"\n  tag {tag_id} is not one of the 8 waypoint tags -- ignored")
                continue
            visible = any(d.tag_id == tag_id for d in detections)
            if not visible:
                print(f"\n  tag {tag_id} not currently visible -- drive/center the robot on it first "
                      f"(this is just a confirmation check, the tag's own reading isn't recorded)")
                continue
            pose = t.get_pose()
            pos = pose.get("position", {})
            x, y = pos.get("x", 0.0), pos.get("y", 0.0)
            recorded[tag_id] = {
                "id": WAYPOINT_TAG_IDS[tag_id],
                "tag_id": tag_id,
                "x": round(x, 3),
                "y": round(y, 3),
            }
            print(f"\n  recorded {WAYPOINT_TAG_IDS[tag_id]} (tag {tag_id}) at get_pose() ({x:.3f}, {y:.3f})")

    cv2.destroyAllWindows()
    if recorded:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(recorded, f, indent=2)
        print(f"\nFinal save: {len(recorded)} measurements -> {OUTPUT_PATH}")
        missing = [wp for tid, wp in WAYPOINT_TAG_IDS.items() if tid not in recorded]
        if missing:
            print(f"NOT recorded this session: {missing} -- measure these by hand if the "
                  f"robot couldn't physically reach them, then hand-edit the JSON.")
        print("\nNext step: these x/y values are ALREADY in get_pose()'s frame -- no manual "
              "conversion needed this time (drive-and-center sidesteps that entirely). Copy "
              "each {\"id\", \"x\", \"y\"} straight into real_feed.py's WAYPOINTS list, replacing "
              "the matching placeholder entry.")


if __name__ == "__main__":
    main()