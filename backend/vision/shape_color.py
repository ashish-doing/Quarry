"""
shape_color.py -- coarse shape + color heuristic for the 8 mission objects.

SCOPE, deliberately narrow (same honesty-about-limits tone as
collision_guard.py's "not real depth" framing -- read that comment if
you haven't):

  - Shape is one of "cone" | "cylinder" | "box" | None. This is a
    silhouette heuristic, NOT a trained classifier. It looks at the
    contour of whatever's brightest-against-background in the crop and
    checks (a) whether it tapers top-to-bottom and (b) how much of its
    own bounding box it fills. That's it. It WILL be wrong from bad
    angles, in poor lighting, or if the crop is mostly background --
    it's a secondary signal to sit alongside OSNet+OCR, not a
    replacement for either.

  - Color is ONLY blue vs pink, because those are the only two colors
    that exist in this mission's object set (see objects_config.json).
    This is intentionally not a general color classifier -- building
    one would be solving a problem we don't have.

Both functions return None on ambiguity or failure. None is not
"wrong" here, it's "no confident signal" -- exactly the same
philosophy as ocr_check.py's "no signal is never an anomaly."

TUNING NOTE: the HSV ranges and shape thresholds below are starting
points, not measured values -- same disclaimer as inner_agent.py's
tuning constants. Verify against real captured crops from the venue
before the demo; lighting changes will shift these.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("quarry.vision.shape_color")

# -- color ranges (OpenCV HSV: H 0-179, S 0-255, V 0-255) -------------------
# Deliberately tight and non-overlapping. If a pixel doesn't clearly fall
# in one of these, it's not counted -- we're not trying to cover the full
# color wheel, just tell blue and pink apart.
_BLUE_HSV_LOW = np.array([95, 80, 50])
_BLUE_HSV_HIGH = np.array([130, 255, 255])

# Pink/magenta wraps toward the low end of hue too (near red), but the
# mission's pink parts are closer to magenta -- keep this range narrow to
# avoid catching red/orange background clutter.
_PINK_HSV_LOW = np.array([145, 60, 80])
_PINK_HSV_HIGH = np.array([175, 255, 255])

MIN_COLOR_FRACTION = 0.08  # at least this fraction of crop pixels must
                             # match a color for it to count as a signal

# -- shape thresholds ---------------------------------------------------
MIN_CONTOUR_AREA_FRACTION = 0.05  # ignore crops where nothing meaningful
                                    # was segmented (mostly background)
TAPER_RATIO_CONE = 0.55   # top-third width / bottom-third width below this
                            # (or its inverse) => tapers => cone
BOX_FILL_RATIO = 0.75     # contour area / bounding-rect area above this,
                            # with low taper => box (fills its rect well,
                            # hard corners)
CYLINDER_FILL_RATIO_MIN = 0.45  # cylinders fill less of their rect than
                                   # boxes (rounded silhouette) but don't
                                   # taper like cones

# Fill ratio alone can't separate box from cylinder -- a photographed
# cylinder's vertical sides are straight (tangent lines), same as a box's,
# so overall bounding-rect fill can come out nearly identical for both.
# The real tell is HOW MANY ROWS IT TAKES from the very top to reach the
# object's full body width: a box is at full width immediately (row 0,
# flat top); a cylinder's curved cap takes a real number of rows to widen
# out to full width. This is measured directly (rows-to-full-width) rather
# than compared at a couple of fixed-fraction rows, because a shallow cap
# can finish curving within just a few percent of the object's height --
# a fixed sample row can land past the curve and look flat either way.
FULL_WIDTH_THRESHOLD = 0.92    # "at full width" = row width >= this fraction of the widest row
FLAT_TOP_ROW_FRACTION_MAX = 0.03  # reaching full width within this fraction of height => flat top => box signal


MAX_CONTOUR_AREA_FRACTION = 0.92  # a contour this close to the full crop
                                     # area is almost certainly the background,
                                     # not the object -- reject it (see the
                                     # polarity-guessing note below)


def _best_contour_for_polarity(thresh: np.ndarray, crop_area: int) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < MIN_CONTOUR_AREA_FRACTION * crop_area or area > MAX_CONTOUR_AREA_FRACTION * crop_area:
        return None
    return largest


def _largest_contour(crop: np.ndarray) -> Optional[np.ndarray]:
    """Segments the crop against its background and returns the largest
    contour, or None if nothing meaningful was found. Assumes (same as
    OSNet's training_crop) the object dominates the frame -- this is NOT
    a general-purpose segmentation, just enough to get a silhouette.

    Otsu's threshold doesn't know which side (above/below threshold) is
    "object" vs "background" -- if the background happens to be brighter
    than the object (e.g. a light wall/floor behind a darker-colored
    part), the naive THRESH_BINARY polarity segments the *background* as
    foreground instead, producing a contour that hugs the whole frame
    border. Guard against that by trying both polarities and rejecting
    whichever result is implausibly large (near the full crop area) --
    that's a background pick, not an object pick."""
    if crop is None or crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    crop_area = crop.shape[0] * crop.shape[1]

    _, thresh_normal = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate = _best_contour_for_polarity(thresh_normal, crop_area)
    if candidate is not None:
        return candidate

    _, thresh_inv = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _best_contour_for_polarity(thresh_inv, crop_area)


def _row_width(mask_row: np.ndarray) -> int:
    """Horizontal extent (rightmost - leftmost foreground pixel) in one
    row of a binary mask. 0 if the row is empty."""
    cols = np.where(mask_row > 0)[0]
    if cols.size == 0:
        return 0
    return int(cols[-1] - cols[0])


def _classify_shape(contour: np.ndarray, crop_shape: Tuple[int, int]) -> Optional[str]:
    x, y, w, h = cv2.boundingRect(contour)
    if w == 0 or h == 0:
        return None

    mask = np.zeros((crop_shape[0], crop_shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    roi = mask[y:y + h, x:x + w]

    third = max(1, h // 3)
    top_band = roi[0:third, :]
    bottom_band = roi[h - third:h, :]

    top_width = max((_row_width(top_band[r]) for r in range(top_band.shape[0])), default=0)
    bottom_width = max((_row_width(bottom_band[r]) for r in range(bottom_band.shape[0])), default=0)

    if top_width == 0 or bottom_width == 0:
        return None

    taper_ratio = min(top_width, bottom_width) / max(top_width, bottom_width)
    fill_ratio = cv2.contourArea(contour) / float(w * h)

    tapers = taper_ratio < TAPER_RATIO_CONE
    if tapers:
        return "cone"

    # Rows-to-full-width check: walk down from the top row and find the
    # first row that reaches (near) the object's full body width. A flat
    # (box) top gets there immediately; a curved (cylinder) cap takes a
    # real number of rows.
    row_widths = [_row_width(roi[r]) for r in range(h)]
    full_width = max(row_widths, default=0)
    rise_row = h  # default: never reaches full width (shouldn't happen, but stay honest)
    if full_width > 0:
        threshold_width = FULL_WIDTH_THRESHOLD * full_width
        for r, rw in enumerate(row_widths):
            if rw >= threshold_width:
                rise_row = r
                break
    has_flat_top = (rise_row / h) <= FLAT_TOP_ROW_FRACTION_MAX

    if has_flat_top and fill_ratio >= BOX_FILL_RATIO:
        return "box"
    if fill_ratio >= CYLINDER_FILL_RATIO_MIN:
        return "cylinder"
    return None  # doesn't confidently match any of the three -- honest null


def _classify_color(crop: np.ndarray) -> Optional[str]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    total = crop.shape[0] * crop.shape[1]
    if total == 0:
        return None

    blue_mask = cv2.inRange(hsv, _BLUE_HSV_LOW, _BLUE_HSV_HIGH)
    pink_mask = cv2.inRange(hsv, _PINK_HSV_LOW, _PINK_HSV_HIGH)

    blue_fraction = float(np.count_nonzero(blue_mask)) / total
    pink_fraction = float(np.count_nonzero(pink_mask)) / total

    if blue_fraction < MIN_COLOR_FRACTION and pink_fraction < MIN_COLOR_FRACTION:
        return None
    return "blue" if blue_fraction >= pink_fraction else "pink"


def detect_shape_and_color(crop: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort (shape, color) read for a captured object crop.

    crop: RGB image (same convention as the rest of the vision/ package --
    matches what's already passed to OSNet/OCR, e.g. candidate.training_crop).

    Returns (shape, color), each independently None if not confidently
    detected. Never raises -- a detection failure here should never take
    down a scan tick; log and return (None, None) instead.
    """
    try:
        if crop is None or crop.size == 0:
            return None, None

        contour = _largest_contour(crop)
        shape = _classify_shape(contour, crop.shape[:2]) if contour is not None else None
        color = _classify_color(crop)
        return shape, color
    except Exception:  # noqa: BLE001 -- best-effort signal, never crash the scan loop
        logger.exception("shape_color: detection failed on this crop")
        return None, None