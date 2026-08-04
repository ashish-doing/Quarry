"""
apriltag_detector.py -- AprilTag pose estimation for QUARRY waypoints.

Uses tag36h11 (matches the printed sheet: IDs 0-7, 100mm). Requires
camera intrinsics from a real calibration pass (see calibrate_camera.py)
-- pose numbers from uncalibrated intrinsics will be wrong in a way that
looks plausible, which is worse than an obvious failure. Same honesty
posture as detector.py: if the intrinsics file is missing, this refuses
to fabricate a pose rather than guessing focal length from FOV metadata.

Tag ID plan (8 tags printed, 6 waypoints):
  IDs 0-5  -> W1-W6 (patrol waypoints, per real_feed.py's WAYPOINTS list)
  IDs 6-7  -> RESERVED as fixed origin/reference markers. Place these two
              at known, fixed points in the venue (e.g. two room corners)
              BEFORE measuring anything else -- they define the world
              coordinate frame that all other waypoint measurements are
              made relative to. Skipping this and eyeballing each
              waypoint's (x, y) independently is how small errors compound
              into a patrol pattern that drifts from what the map shows.

Usage:
    detector = AprilTagDetector()
    detections = detector.detect(frame)  # frame: HxWx3 numpy array
    for d in detections:
        print(d.tag_id, d.distance_m, d.x_m, d.y_m)
"""

import json
import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger("quarry.vision.apriltag")

TAG_SIZE_M = 0.100  # 100mm, matches the printed sheet -- change here if you reprint at a different size
WAYPOINT_TAG_IDS = {0: "W1", 1: "W2", 2: "W3", 3: "W4", 4: "W5", 5: "W6"}
REFERENCE_TAG_IDS = {6: "origin_A", 7: "origin_B"}

DEFAULT_INTRINSICS_PATH = os.path.join(os.path.dirname(__file__), "camera_intrinsics.json")


class AprilTagDetection:
    __slots__ = ("tag_id", "x_m", "y_m", "z_m", "distance_m", "corners")

    def __init__(self, tag_id, translation, corners):
        self.tag_id = tag_id
        # translation is [x, y, z] in the CAMERA's frame, meters, from the
        # detector's own solvePnP-equivalent -- z is depth (distance along
        # the optical axis), x/y are lateral offset. This is NOT yet in
        # room/world coordinates -- that transform needs the two reference
        # tags placed and measured first (see module docstring).
        self.x_m = float(translation[0])
        self.y_m = float(translation[1])
        self.z_m = float(translation[2])
        self.distance_m = float(np.linalg.norm(translation))
        self.corners = corners

    def __repr__(self):
        return f"AprilTagDetection(id={self.tag_id}, dist={self.distance_m:.2f}m)"


class AprilTagDetector:
    def __init__(self, intrinsics_path: str = DEFAULT_INTRINSICS_PATH, tag_size_m: float = TAG_SIZE_M):
        self.tag_size_m = tag_size_m
        self.intrinsics = self._load_intrinsics(intrinsics_path)
        self.detector = None
        self._load_backend()

    def _load_intrinsics(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            logger.warning(
                "No camera_intrinsics.json found at %s -- run calibrate_camera.py "
                "FIRST. Pose estimates are disabled until real intrinsics exist; "
                "a guessed fx/fy from FOV metadata would produce plausible-looking "
                "but wrong distances, which is worse than refusing.", path,
            )
            return None
        with open(path) as f:
            data = json.load(f)
        required = ("fx", "fy", "cx", "cy")
        if not all(k in data for k in required):
            logger.error("camera_intrinsics.json is missing required keys %s", required)
            return None
        return data

    def _load_backend(self):
        try:
            from pupil_apriltags import Detector

            self.detector = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
            logger.info("AprilTag detector loaded (tag36h11)")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pupil-apriltags unavailable (%s) -- `pip install pupil-apriltags`. "
                "AprilTag detection disabled.", exc,
            )
            self.detector = None

    @property
    def available(self) -> bool:
        return self.detector is not None and self.intrinsics is not None

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        import cv2
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def detect(self, frame: np.ndarray) -> List[AprilTagDetection]:
        if not self.available:
            return []
        gray = self._to_grayscale(frame)
        cam_params = (
            self.intrinsics["fx"], self.intrinsics["fy"],
            self.intrinsics["cx"], self.intrinsics["cy"],
        )
        try:
            results = self.detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=cam_params, tag_size=self.tag_size_m,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("AprilTag detection failed: %s", exc)
            return []

        detections = []
        for r in results:
            # pose_t is a 3x1 translation vector, camera frame, meters
            translation = r.pose_t.flatten()
            detections.append(AprilTagDetection(r.tag_id, translation, r.corners))
        return detections

    @staticmethod
    def label_for(tag_id: int) -> str:
        if tag_id in WAYPOINT_TAG_IDS:
            return WAYPOINT_TAG_IDS[tag_id]
        if tag_id in REFERENCE_TAG_IDS:
            return REFERENCE_TAG_IDS[tag_id]
        return f"unknown_tag_{tag_id}"