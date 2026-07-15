"""
reid.py -- appearance-based re-identification for QUARRY.

Turns a cropped detection into a fixed-length embedding so a
registered target can be recognised even after an appearance change
(different jacket, hat, etc.) -- this is the "candidate re-id
approach" from Section 8 of the master doc: OSNet primary, with the
door left open for a CLIP-family fallback if OSNet proves too heavy
alongside YOLOE-26 on the shared 6GB VRAM budget.

Usage:
    matcher = TargetMatcher()
    matcher.register("intruder-01", registration_photo)   # numpy array
    name, score = matcher.best_match(cropped_detection)
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("quarry.vision.reid")

MATCH_THRESHOLD = 0.62  # cosine similarity -- tune against real captures, don't trust blindly


class TargetMatcher:
    def __init__(self, backend: str = "osnet"):
        self.backend = backend
        self._embeddings: Dict[str, np.ndarray] = {}
        self._extractor = None
        self._load_backend()

    def _load_backend(self):
        try:
            if self.backend == "osnet":
                # torchreid ships several OSNet variants; osnet_x0_25 is the
                # smallest -- realistic to run alongside YOLOE-26 on the
                # RTX 4050's 6GB without fighting it for VRAM.
                from torchreid.reid.utils import FeatureExtractor

                self._extractor = FeatureExtractor(
                    model_name="osnet_x0_25",
                    model_path="",  # empty -> downloads pretrained weights automatically
                    device="cuda",
                )
                logger.info("OSNet (osnet_x0_25) re-id backend loaded")
            else:
                raise ValueError(f"unknown backend '{self.backend}'")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Re-id backend unavailable (%s) -- registered targets will NOT be "
                "matched by appearance, only ever flagged as generic detections. "
                "Check `pip install torchreid` and that torch has CUDA available.", exc,
            )
            self._extractor = None

    @property
    def available(self) -> bool:
        return self._extractor is not None

    def _embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self._extractor is None or image is None or image.size == 0:
            return None
        try:
            feats = self._extractor([image])
            vec = feats[0].cpu().numpy() if hasattr(feats[0], "cpu") else np.asarray(feats[0])
            norm = np.linalg.norm(vec) + 1e-8
            return vec / norm
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding extraction failed: %s", exc)
            return None

    def register(self, name: str, image: np.ndarray):
        vec = self._embed(image)
        if vec is not None:
            self._embeddings[name] = vec
            logger.info("Registered target '%s' with a %d-dim embedding", name, vec.shape[0])
        else:
            logger.warning(
                "Target '%s' registered WITHOUT an embedding (backend unavailable or bad "
                "photo) -- it can only be confirmed by manual team voting, never "
                "auto-matched by appearance.", name,
            )

    def best_match(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Returns (best_matching_name, score) if score clears
        MATCH_THRESHOLD, else (None, best_score_seen)."""
        if not self._embeddings:
            return None, 0.0
        vec = self._embed(crop)
        if vec is None:
            return None, 0.0

        best_name, best_score = None, -1.0
        for name, ref in self._embeddings.items():
            score = float(np.dot(vec, ref))
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score