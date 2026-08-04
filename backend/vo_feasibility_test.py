"""
vo_feasibility_test.py -- monocular visual-odometry feasibility check.

THIS IS A GO/NO-GO EXPERIMENT, NOT A NAVIGATION SYSTEM. Time-boxed on
purpose: a feature-poor room (confirmed earlier as the likely case for
the demo venue) is close to the worst-case input for monocular tracking
-- blank walls and uniform floor starve ORB of anything to lock onto,
and monocular VO has an unrecoverable scale ambiguity even in good
conditions (this is exactly why AprilTags exist in this project -- they
give a known real-world size to resolve scale, which VO alone cannot).

What this script does: tracks ORB features frame-to-frame using optical
flow, logs feature count and a rough frame-to-frame motion estimate, and
tells you plainly whether tracking held or fell apart. If it fails here,
in your own room, it will fail worse in an empty venue -- that's the
signal to cut VO from scope, not push harder on it.

Usage:
    python backend/vo_feasibility_test.py --source webcam   # any room, right now
    python backend/vo_feasibility_test.py --source twin     # against the real UGV camera

Controls: 'q' to quit and print the summary verdict.
"""

import argparse
import os
import time

import cv2
import numpy as np

MIN_FEATURES_HEALTHY = 60     # below this, tracking is considered unreliable for this tick
MIN_MATCH_RATIO = 0.35        # fraction of prev-frame features that must survive to next frame
LOG_INTERVAL_S = 0.5


def _frame_source(source: str):
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
        from dotenv import load_dotenv
        import cyberwave

        load_dotenv()
        if "CYBERWAVE_API_KEY" not in os.environ:
            raise RuntimeError("CYBERWAVE_API_KEY not set -- add it to backend/.env")
        cyberwave.configure(api_key=os.environ["CYBERWAVE_API_KEY"])
        env_id = os.environ.get("CYBERWAVE_ENVIRONMENT_ID", "9383b6b2-0df7-4e99-8d9e-8a352eb6e1ab")
        t = cyberwave.twin("waveshare/ugv-beast", environment=env_id, name="QUARRY")
        while True:
            frame = t.get_frame("numpy", source="remote_edge")
            if frame is None:
                continue
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"unknown source '{source}'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["webcam", "twin"], default="webcam")
    args = parser.parse_args()

    orb = cv2.ORB_create(nfeatures=500)
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    prev_gray, prev_pts = None, None
    tick_log = []  # (timestamp, feature_count, match_ratio)
    last_log = 0.0

    print(f"VO feasibility test against source='{args.source}'. Move the camera around "
          f"naturally -- pan, walk a bit, the way the robot actually would. 'q' to finish.")

    for frame in _frame_source(args.source):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = frame.copy()

        if prev_gray is None:
            kp = orb.detect(gray, None)
            prev_pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)
            prev_gray = gray
            continue

        if prev_pts is not None and len(prev_pts) > 0:
            next_pts, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            good_new = next_pts[status.flatten() == 1] if next_pts is not None else np.array([])
            good_old = prev_pts[status.flatten() == 1] if next_pts is not None else np.array([])
            match_ratio = len(good_new) / max(1, len(prev_pts))

            for pt in good_new:
                x, y = pt.ravel()
                cv2.circle(display, (int(x), int(y)), 3, (0, 255, 0), -1)

            now = time.monotonic()
            if now - last_log >= LOG_INTERVAL_S:
                last_log = now
                tick_log.append((now, len(good_new), match_ratio))
                status_str = "OK" if len(good_new) >= MIN_FEATURES_HEALTHY and match_ratio >= MIN_MATCH_RATIO else "WEAK"
                print(f"\r  tracked={len(good_new):3d}  match_ratio={match_ratio:.2f}  [{status_str}]   ",
                      end="", flush=True)

            # Re-detect periodically so feature count doesn't just monotonically decay
            if len(good_new) < MIN_FEATURES_HEALTHY:
                kp = orb.detect(gray, None)
                prev_pts = np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 1, 2)
            else:
                prev_pts = good_new.reshape(-1, 1, 2)

        prev_gray = gray
        cv2.putText(display, "VO feasibility -- 'q' to finish", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("vo_feasibility", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    if not tick_log:
        print("\nNo data collected -- camera source may not have produced frames.")
        return

    feature_counts = [t[1] for t in tick_log]
    match_ratios = [t[2] for t in tick_log]
    healthy_ticks = sum(
        1 for _, fc, mr in tick_log if fc >= MIN_FEATURES_HEALTHY and mr >= MIN_MATCH_RATIO
    )
    healthy_fraction = healthy_ticks / len(tick_log)

    print(f"\n\n--- VO feasibility verdict ({len(tick_log)} ticks, source='{args.source}') ---")
    print(f"  Mean tracked features: {np.mean(feature_counts):.0f}  (min {min(feature_counts)}, max {max(feature_counts)})")
    print(f"  Mean match ratio:      {np.mean(match_ratios):.2f}")
    print(f"  Healthy ticks:         {healthy_fraction:.0%}")

    if healthy_fraction >= 0.7:
        print(f"\n  VERDICT: tracking held up reasonably well here. Still only a relative-motion")
        print(f"  signal (no real-world scale without AprilTags) -- worth a supplementary")
        print(f"  experiment, not a primary nav system, given the time budget.")
    elif healthy_fraction >= 0.3:
        print(f"\n  VERDICT: marginal. Tracking degrades often enough that it's not reliable")
        print(f"  standalone. If the venue room is more feature-poor than here, expect worse.")
    else:
        print(f"\n  VERDICT: tracking failed most of the time. This matches the expected")
        print(f"  worst-case for monocular VO in a sparse room. Recommend cutting VO from")
        print(f"  scope now and relying on AprilTag waypoint gating + coverage-tracking sweep,")
        print(f"  same honest fallback pattern patrol.py already uses for yaw.")


if __name__ == "__main__":
    main()