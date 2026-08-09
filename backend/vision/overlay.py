"""
overlay.py -- shared annotated-video-feed drawing helpers.

Extracted out of site_agent.py so inner_agent.py (the new merged
driving+relay process for the inner UGV) can use the exact same
per-label coloring and cube-outline rendering without a second copy
drifting out of sync. Pure OpenCV drawing, no state, no I/O -- safe to
import from either process.
"""

import cv2

# One fixed, high-contrast color per label so different object types are
# instantly distinguishable in the live feed, instead of everything being
# the same green. BGR tuples (OpenCV convention, not RGB). Cycles by hash
# if a label isn't in this explicit map, so new vocabulary entries still
# get a stable, distinct-ish color without needing a manual update here.
_LABEL_COLORS = {
    "person":    (0, 0, 255),     # red -- highest-attention label
    "clothing":  (128, 128, 128), # gray -- deliberately dull, the "false alarm" label
    "backpack":  (0, 255, 255),   # yellow
    "bag":       (0, 200, 255),   # orange-yellow
    "box":       (255, 128, 0),   # blue-ish
    "toolbox":   (255, 0, 128),   # purple-ish
    "chair":     (0, 255, 0),     # green
}
_FALLBACK_PALETTE = [
    (255, 0, 0), (0, 128, 255), (255, 0, 255), (128, 255, 0),
    (0, 200, 200), (200, 200, 0), (180, 0, 180),
]


def color_for_label(label: str):
    if label in _LABEL_COLORS:
        return _LABEL_COLORS[label]
    return _FALLBACK_PALETTE[hash(label) % len(_FALLBACK_PALETTE)]


# ---------------------------------------------------------------------------
# Cube-outline overlay -- per the master doc's locked decision: "stylized
# 2D overlay ONLY, not real depth." This hardware has no lidar/depth
# camera, so there is nothing real to draw a 3D box from. What follows is
# a fixed-fraction visual flourish -- a "back face" offset by a constant
# percentage of the box's own width/height, connected to the real 2D bbox
# corners -- for pitch-material polish only. NEVER read this as, or
# present it as, an actual 3D reconstruction.
# ---------------------------------------------------------------------------
CUBE_DEPTH_FRACTION = 0.18


def draw_cube_overlay(frame, x1: int, y1: int, x2: int, y2: int, color):
    """Draws a pseudo-3D wireframe box over an existing 2D bbox. Purely a
    rendering flourish (see module note above)."""
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    dx, dy = max(2, int(w * CUBE_DEPTH_FRACTION)), max(2, int(h * CUBE_DEPTH_FRACTION))
    bx1, by1, bx2, by2 = x1 + dx, y1 - dy, x2 + dx, y2 - dy

    thin = 1
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thin)
    for (fx, fy), (bxp, byp) in (
        ((x1, y1), (bx1, by1)), ((x2, y1), (bx2, by1)),
        ((x2, y2), (bx2, by2)), ((x1, y2), (bx1, by2)),
    ):
        cv2.line(frame, (fx, fy), (bxp, byp), color, thin)


def draw_detections(frame, detections):
    """Draw a box + label/confidence + pseudo-3D cube overlay per
    detection, one distinct color per label. Returns a new annotated
    frame (does not mutate the input)."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        color = color_for_label(det.label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        draw_cube_overlay(annotated, x1, y1, x2, y2, color)
        label = f"{det.label} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return annotated