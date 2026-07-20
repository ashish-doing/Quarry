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
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("quarry.vision.reid")

MATCH_THRESHOLD = 0.62  # cosine similarity -- tune against real captures, don't trust blindly


class TargetMatcher:
    def __init__(self, backend: str = "osnet"):
        self.backend = backend
        self._embeddings: Dict[str, List[np.ndarray]] = {}  # name -> gallery of embeddings
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

    def register(self, name: str, images):
        """images: a single HxWx3 array, or a list of them. Register from
        3-5 angles for real re-id robustness (see the perception hardening
        notes) -- one photo still works but is the weaker case. Each valid
        photo adds one embedding to that target's gallery. Matching takes
        the BEST similarity across the whole gallery, not an average --
        averaging would blur away the exact appearance variation that
        makes multi-angle registration worth doing in the first place."""
        if isinstance(images, np.ndarray):
            images = [images]
        gallery = self._embeddings.setdefault(name, [])
        added = 0
        for image in images:
            vec = self._embed(image)
            if vec is not None:
                gallery.append(vec)
                added += 1
        if added:
            logger.info("Registered target '%s' with %d/%d new embedding(s), gallery size now %d",
                        name, added, len(images), len(gallery))
        else:
            logger.warning(
                "Target '%s' registered WITHOUT any embeddings (backend unavailable or all "
                "photos bad) -- it can only be confirmed by manual team voting, never "
                "auto-matched by appearance.", name,
            )

    def best_match(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Returns (best_matching_name, score) if score clears
        MATCH_THRESHOLD, else (None, best_score_seen). Score against a
        target is the max similarity across its whole photo gallery."""
        if not self._embeddings:
            return None, 0.0
        vec = self._embed(crop)
        if vec is None:
            return None, 0.0

        best_name, best_score = None, -1.0
        for name, gallery in self._embeddings.items():
            if not gallery:
                continue
            score = max(float(np.dot(vec, ref)) for ref in gallery)
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= MATCH_THRESHOLD:
            return best_name, best_score
        return None, best_score