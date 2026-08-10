"""
reid.py -- appearance-based re-identification for QUARRY.

Turns a cropped detection into a fixed-length embedding so a
registered target can be recognised even after an appearance change
(different jacket, hat, etc.) -- OSNet primary, with the door left open
for a CLIP-family fallback if OSNet proves too heavy alongside YOLOE-26
on the shared 6GB VRAM budget.

Usage:
    matcher = TargetMatcher()
    matcher.register("object-01", registration_photo)   # numpy array
    name, score = matcher.best_match(cropped_detection)

Gallery persistence (added for the mission-object registration flow --
register_mission_objects.py registers all 8 objects offline, in its own
short-lived process, then saves the gallery; inner_agent.py's
RealRobotState loads it back at startup. Without this, embeddings built
in one process would never reach another -- register_mission_objects.py
and inner_agent.py are always separate processes.):

    matcher.save_gallery()                  # after registering everything
    matcher.load_gallery()                  # called automatically by
                                             # RealRobotState.__init__
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("quarry.vision.reid")

MATCH_THRESHOLD = 0.62  # cosine similarity -- tune against real captures, don't trust blindly

# Default gallery file location -- one level up from backend/vision/, so it
# sits at backend/target_gallery.pkl, next to objects_config.json. Pickle
# rather than npz because the gallery is a dict of variable-length embedding
# lists per target, not a fixed-shape array -- npz would need padding/
# ragged-array workarounds for no real benefit here.
GALLERY_DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "..", "target_gallery.pkl")


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
        3-5 angles for real re-id robustness -- one photo still works but
        is the weaker case. Each valid photo adds one embedding to that
        target's gallery. Matching takes the BEST similarity across the
        whole gallery, not an average -- averaging would blur away the
        exact appearance variation that makes multi-angle registration
        worth doing in the first place."""
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

    def save_gallery(self, path: str = GALLERY_DEFAULT_PATH) -> bool:
        """Persists the current embedding gallery to disk (pickle of
        {name: [np.ndarray, ...]}). Call this once after registering all
        mission objects, from register_mission_objects.py -- embeddings
        built in a short-lived registration process are otherwise lost
        the moment that process exits. Returns False (and logs, doesn't
        raise) on failure -- a failed save shouldn't crash a registration
        script that a person is running interactively and watching."""
        if not self._embeddings:
            logger.warning("save_gallery() called with an empty gallery -- nothing to save.")
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self._embeddings, f)
            total = sum(len(g) for g in self._embeddings.values())
            logger.info("Saved gallery: %d target(s), %d total embedding(s) -> %s",
                        len(self._embeddings), total, path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to save gallery to %s: %s", path, exc)
            return False

    def load_gallery(self, path: str = GALLERY_DEFAULT_PATH, merge: bool = False) -> bool:
        """Loads a previously-saved gallery from disk. By default REPLACES
        the current in-memory gallery (merge=False) -- the normal startup
        case, where a fresh RealRobotState has an empty gallery and just
        wants the saved one. Pass merge=True to add to whatever's already
        registered instead of replacing it. Missing file is not an error
        (a fresh install with no gallery yet is a completely normal,
        expected state) -- logs at info level and returns False, doesn't
        raise or warn."""
        if not os.path.exists(path):
            logger.info("No saved gallery found at %s -- starting with an empty one "
                        "(run register_mission_objects.py before a mission if this is "
                        "unexpected).", path)
            return False
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            if not isinstance(loaded, dict):
                logger.error("Gallery file %s did not contain the expected dict shape -- ignoring.", path)
                return False
            if merge:
                for name, embeddings in loaded.items():
                    self._embeddings.setdefault(name, []).extend(embeddings)
            else:
                self._embeddings = loaded
            total = sum(len(g) for g in self._embeddings.values())
            logger.info("Loaded gallery: %d target(s), %d total embedding(s) <- %s",
                        len(self._embeddings), total, path)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load gallery from %s: %s", path, exc)
            return False