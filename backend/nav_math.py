"""
nav_math.py -- pure geometry helpers for the sequential-survey nav
controller (sequential_patrol.py). Extracted into their own module
specifically so they're unit-testable without importing real_feed.py
(which pulls in cyberwave/YOLOE/OSNet at module level) -- this was a
real, acknowledged gap flagged in test_sequential_survey_components.py's
own docstring; this module closes it.

No hardware, no numpy even -- just tuples of floats and math.*, so this
has zero import risk on any machine, CI runner, or teammate's laptop.
"""

import math
from typing import Optional, Tuple

Vec2 = Tuple[float, float]


def vector_to(from_xy: Vec2, to_xy: Vec2) -> Vec2:
    """Vector FROM from_xy TO to_xy."""
    return (to_xy[0] - from_xy[0], to_xy[1] - from_xy[1])


def normalize(v: Vec2) -> Optional[Vec2]:
    """Unit vector in the direction of v, or None if v is effectively the
    zero vector (can't derive a direction from no movement -- this is the
    honest "no signal" case, not a division-by-zero to paper over with a
    fallback direction)."""
    mag = math.hypot(*v)
    if mag < 1e-9:
        return None
    return (v[0] / mag, v[1] / mag)


def signed_angle_between(v1: Vec2, v2: Vec2) -> float:
    """Signed angle FROM v1 TO v2, in radians, positive = counter-clockwise
    (turn left), negative = clockwise (turn right). Standard atan2(cross,
    dot) formulation -- avoids the sign ambiguity a plain dot-product-based
    angle (acos) would have, which matters here since the controller needs
    to know WHICH way to turn, not just how far off it is."""
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.atan2(cross, dot)


def waypoint_xy(waypoint_id: str, waypoints: list) -> Optional[Vec2]:
    """Looks up a waypoint's (x, y) from a WAYPOINTS-shaped list (list of
    dicts with at least "id", "x", "y" keys -- matches real_feed.WAYPOINTS'
    actual shape, but takes it as a plain argument rather than importing
    real_feed, keeping this module's zero-import-risk property intact)."""
    for wp in waypoints:
        if wp["id"] == waypoint_id:
            return (wp["x"], wp["y"])
    return None


def location_offset(waypoint_id: str, position: dict, waypoints: list) -> Optional[dict]:
    """Waypoint + offset location (Section 5 Task 6's locked decision) --
    a cheap vector subtraction against the object's nearest known waypoint.
    Identical math to real_feed.py's _location_offset_for(); this is the
    canonical, tested version -- real_feed.py and sequential_patrol.py
    both should eventually import this instead of each keeping their own
    copy, closing the "two copies of the same formula" risk flagged
    earlier rather than just tolerating it."""
    wp_xy = waypoint_xy(waypoint_id, waypoints)
    if wp_xy is None:
        return None
    return {
        "waypoint": waypoint_id,
        "offset_x": round(position["x"] - wp_xy[0], 3),
        "offset_y": round(position["y"] - wp_xy[1], 3),
    }