"""
detector.py -- YOLOE-26 wrapper for QUARRY.

Loads Cyberwave's Edge-tagged YOLOE-26 checkpoint (open-vocabulary,
real-time detector) and exposes a single detect() call that returns
plain (label, confidence, bbox) objects -- nothing model-specific
leaks past this file, so swapping models later only touches this
module, not real_feed.py or the WebSocket contract.

NOTE ON THE CHECKPOINT: verify the exact weights path/name against
Cyberwave's AI Models catalog (cyberwave.com -> AI Models -> filter
"Edge") before first run. YOLOE ships via the ultralytics package as
of their 2025 "Real-Time Seeing Anything" release; the loader below
assumes that API shape but you should confirm the exact class name
and .set_classes() signature against whatever version pip resolves --
these open-vocabulary APIs move fast and may have changed since.

If the model can't be loaded (missing weights, no GPU, wrong API),
detect() degrades to returning an empty list rather than crashing the
whole feed -- a dead detector shouldn't take the dashboard down with
it. Check logs for "detection disabled" if candidates stop appearing.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger("quarry.vision.detector")

BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2, normalized 0-1


class Detection:
    __slots__ = ("label", "confidence", "bbox")

    def __init__(self, label: str, confidence: float, bbox: BBox):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox

    def __repr__(self):
        return f"Detection({self.label!r}, conf={self.confidence:.2f})"


class YoloeDetector:
    DEFAULT_VOCABULARY = ["person", "backpack", "chair", "toolbox", "box", "bag"]

    def __init__(
        self,
        weights: str = "yoloe-11s-seg.pt",
        confidence_threshold: float = 0.45,
        device: str = "cuda",
        vocabulary: Optional[List[str]] = None,
    ):
        self.confidence_threshold = confidence_threshold
        self.vocabulary = vocabulary or self.DEFAULT_VOCABULARY
        self.device = device
        self.model = None
        self._load(weights)

    def _load(self, weights: str):
        try:
            from ultralytics import YOLOE

            self.model = YOLOE(weights)
            if hasattr(self.model, "set_classes") and self.vocabulary:
                # open-vocabulary prompt -- restrict detection to labels QUARRY cares about
                self.model.set_classes(self.vocabulary, self.model.get_text_pe(self.vocabulary))
            logger.info("YOLOE-26 loaded from %s on %s (vocabulary=%s)",
                        weights, self.device, self.vocabulary)
        except Exception as exc:  # noqa: BLE001 -- any failure here must not be fatal
            logger.warning(
                "Could not load YOLOE-26 (%s) -- detection disabled, no candidates "
                "will be raised until this is fixed. Check the weights path and that "
                "`pip install ultralytics` pulled a version with YOLOE support.", exc,
            )
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None

    @staticmethod
    def normalize_frame(frame):
        """Turns whatever get_latest_frame() handed back into a plain numpy
        array, or None if it's not something we can use. Exposed as its own
        method so callers outside detect() (like site_agent.py's video
        overlay) can normalize once and reuse the result, instead of detect()
        silently normalizing its own local copy and throwing it away."""
        import numpy as np
        if hasattr(frame, "convert"):  # PIL Image
            return np.array(frame.convert("RGB"))
        if isinstance(frame, np.ndarray):
            return frame
        return None

    def detect(self, frame) -> List[Detection]:
        """frame: HxWx3 numpy array (RGB or BGR, matching whatever the
        camera streamer hands back). Returns detections above threshold."""
        if self.model is None:
            return []

        frame = self.normalize_frame(frame)
        if frame is None:
            logger.debug("Unrecognized frame type -- skipping this tick")
            return []

        try:
            results = self.model.predict(
                frame, device=self.device, verbose=False, conf=self.confidence_threshold
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLOE inference failed: %s", exc)
            return []

        detections: List[Detection] = []
        h, w = frame.shape[:2]
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            names = getattr(self.model, "names", {})
            for box in boxes:
                cls_id = int(box.cls[0])
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(Detection(label, conf, (x1 / w, y1 / h, x2 / w, y2 / h)))
        return detections