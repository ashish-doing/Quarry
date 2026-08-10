"""
ocr_check.py -- OCR cross-check for numbered cardboard objects.

Per the master doc's locked decision (Section 4, "Object identity"):
OSNet re-id is the PRIMARY signal (reused, tested code via TargetMatcher).
This module is the cross-check on top of that, not a replacement for it --
if OCR disagrees with what OSNet already matched, that's a flagged
anomaly for the sightings panel, not an auto-override. Two independent
signals, neither one trusted blindly.

Requires: pip install pytesseract, plus the Tesseract binary itself
installed on the machine (not a pip-installable dependency -- see
README note below). If either is missing, this degrades to "OCR
unavailable" rather than crashing candidate-raising.

Usage:
    checker = MarkerOCR()
    result = checker.read_number(crop)      # crop: HxWx3 numpy array
    # result.number is e.g. "01", or None if nothing readable
    # result.raw_text is the untouched Tesseract output, for debugging

    anomaly = checker.cross_check(result, expected_number="01")
    # anomaly is None if they agree (or OCR had nothing to say),
    # else a short string describing the mismatch for the sightings panel
"""

import logging
import os
import platform
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("quarry.vision.ocr_check")

# Common Windows install locations for the UB-Mannheim Tesseract build.
# pytesseract only finds the binary via PATH -- on Windows that's easy to
# get wrong (the installer doesn't always add it, or a PATH change made
# via SetEnvironmentVariable needs a fresh terminal to take effect, which
# is easy to miss under demo-day time pressure). Tried ONLY as a fallback
# if the binary isn't already resolvable via PATH -- this never overrides
# a working PATH-based setup, and the platform check means it's a no-op
# on Linux/Mac.
_WINDOWS_FALLBACK_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]

# Tesseract's page-segmentation mode 7 = "treat the image as a single text
# line" -- the right mode for a bold, isolated marker digit rather than a
# full page of text. --psm 6 (block of text) is the fallback if 7 reads
# nothing, since a slightly angled/cropped marker can look more like a
# small block than a clean single line.
_TESS_CONFIG_PRIMARY = "--psm 7 -c tessedit_char_whitelist=0123456789"
_TESS_CONFIG_FALLBACK = "--psm 6 -c tessedit_char_whitelist=0123456789"

_NUMBER_RE = re.compile(r"\d+")


@dataclass
class OCRResult:
    number: Optional[str]     # zero-padded to 2 digits if a plausible marker number was read, else None
    raw_text: str              # untouched Tesseract output, for debugging misreads
    confidence: float          # Tesseract's own mean word confidence, 0-100, 0.0 if unavailable


class MarkerOCR:
    def __init__(self):
        self._pytesseract = None
        self._load()

    def _load(self):
        try:
            import pytesseract  # noqa: F401
            self._pytesseract = pytesseract
            # Fail fast and loud (once, at load time) if the Tesseract binary
            # itself isn't on PATH -- pytesseract installs fine via pip even
            # when the actual OCR engine binary is missing, which would
            # otherwise surface as a confusing per-frame exception later.
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR available (%s)", pytesseract.get_tesseract_version())
            return
        except Exception as exc:  # noqa: BLE001
            path_exc = exc

        # PATH resolution failed -- on Windows, try the common UB-Mannheim
        # install locations directly before giving up. This is a fallback
        # ONLY: it doesn't run at all on other platforms, and it only
        # fires because the PATH-based attempt above already failed.
        if platform.system() == "Windows":
            for candidate in _WINDOWS_FALLBACK_PATHS:
                if os.path.isfile(candidate):
                    try:
                        self._pytesseract.pytesseract.tesseract_cmd = candidate
                        self._pytesseract.get_tesseract_version()
                        logger.info(
                            "Tesseract OCR available via fallback path %s "
                            "(not found on PATH -- consider fixing PATH so this "
                            "fallback isn't relied on long-term)", candidate,
                        )
                        return
                    except Exception:  # noqa: BLE001 -- try the next candidate
                        continue

        logger.warning(
            "Tesseract OCR unavailable (%s) -- OCR cross-check disabled, "
            "candidates will be raised on OSNet match alone with no anomaly "
            "flagging. On Windows: install the Tesseract binary separately "
            "(UB-Mannheim build is the common choice) and either add it to "
            "PATH or set pytesseract.pytesseract.tesseract_cmd explicitly.",
            path_exc,
        )
        self._pytesseract = None

    @property
    def available(self) -> bool:
        return self._pytesseract is not None

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Bold, high-contrast marker digits on cardboard -- a simple
        grayscale + Otsu threshold is the right amount of preprocessing
        here. Deliberately not doing anything fancier (deskew, perspective
        correction) since that's real, untested work for a 2-3 day budget
        and the marker digits are specified as bold/high-contrast precisely
        so a plain threshold should be enough."""
        import cv2

        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        # Upscale small crops -- Tesseract reads small/blurry text badly,
        # and a numbered-object crop is often a small fraction of the frame.
        h, w = gray.shape[:2]
        if max(h, w) < 200:
            scale = 200 / max(h, w)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def read_number(self, crop: np.ndarray) -> OCRResult:
        if not self.available or crop is None or crop.size == 0:
            return OCRResult(number=None, raw_text="", confidence=0.0)

        try:
            processed = self._preprocess(crop)
        except Exception as exc:  # noqa: BLE001
            logger.debug("OCR preprocessing failed (%s) -- skipping this tick", exc)
            return OCRResult(number=None, raw_text="", confidence=0.0)

        for config in (_TESS_CONFIG_PRIMARY, _TESS_CONFIG_FALLBACK):
            try:
                text = self._pytesseract.image_to_string(processed, config=config).strip()
            except Exception as exc:  # noqa: BLE001 -- a bad read must not crash candidate-raising
                logger.debug("Tesseract read failed (%s)", exc)
                continue

            match = _NUMBER_RE.search(text)
            if match:
                number = match.group(0).zfill(2)
                confidence = self._mean_word_confidence(processed, config)
                return OCRResult(number=number, raw_text=text, confidence=confidence)

        # Nothing readable in either pass -- honest "no signal", not a guess.
        return OCRResult(number=None, raw_text="", confidence=0.0)

    def _mean_word_confidence(self, processed: np.ndarray, config: str) -> float:
        try:
            data = self._pytesseract.image_to_data(
                processed, config=config, output_type=self._pytesseract.Output.DICT
            )
            confs = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
            return sum(confs) / len(confs) if confs else 0.0
        except Exception:  # noqa: BLE001 -- confidence is a nice-to-have, never fatal
            return 0.0

    @staticmethod
    def cross_check(result: OCRResult, expected_number: str) -> Optional[str]:
        """Returns None if OCR agrees with the OSNet-matched target's expected
        number, OR if OCR simply had nothing readable (no signal is not a
        contradiction -- don't flag an anomaly just because the marker was
        at a bad angle this frame). Returns a short description string if
        OCR read a DIFFERENT number than expected -- that's the real
        anomaly worth surfacing in the sightings panel."""
        if result.number is None:
            return None
        expected = str(expected_number).zfill(2)
        if result.number == expected:
            return None
        return f"OCR read '{result.number}' but OSNet matched object-{expected}"