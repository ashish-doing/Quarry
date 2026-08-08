"""
test_nav_math.py -- real test coverage for the sequential-survey nav
controller's pure geometry (nav_math.py). This is the piece
test_sequential_survey_components.py's own docstring flagged as an
honest gap ("real_feed.py's _location_offset_for() ... NOT covered
here") -- closed by extracting the math into nav_math.py, which has
zero heavy imports and can be tested directly.

Run:
    python -m pytest backend/test_nav_math.py -v
"""

import math

import pytest

from backend.nav_math import (
    vector_to, normalize, signed_angle_between, waypoint_xy, location_offset,
)


# ---------------------------------------------------------------------------
# vector_to / normalize
# ---------------------------------------------------------------------------

def test_vector_to_basic():
    assert vector_to((0, 0), (3, 4)) == (3, 4)
    assert vector_to((1, 1), (1, 1)) == (0, 0)


def test_normalize_returns_unit_vector():
    v = normalize((3, 4))
    assert v is not None
    assert math.isclose(math.hypot(*v), 1.0, abs_tol=1e-9)
    assert math.isclose(v[0], 0.6, abs_tol=1e-9)
    assert math.isclose(v[1], 0.8, abs_tol=1e-9)


def test_normalize_zero_vector_returns_none():
    """No real movement between two ticks means no real heading signal --
    this must return None, not a fabricated direction like (0, 0) or (1, 0),
    since the controller's bootstrap logic depends on distinguishing 'no
    heading yet' from 'a valid but arbitrary heading'."""
    assert normalize((0, 0)) is None


def test_normalize_below_noise_floor_still_returns_a_direction():
    """normalize() itself doesn't apply a noise floor -- that's the
    caller's job (MOVEMENT_NOISE_FLOOR in sequential_patrol.py). A tiny
    but nonzero vector should still normalize cleanly; the noise-floor
    decision of whether to TRUST that direction happens one layer up."""
    v = normalize((0.001, 0))
    assert v is not None
    assert math.isclose(v[0], 1.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# signed_angle_between
# ---------------------------------------------------------------------------

def test_signed_angle_between_same_direction_is_zero():
    assert math.isclose(signed_angle_between((1, 0), (1, 0)), 0.0, abs_tol=1e-9)


def test_signed_angle_between_left_turn_is_positive():
    """Rotating 90 degrees counter-clockwise from facing +x to facing +y
    must read as a POSITIVE angle -- this sign convention is what
    sequential_patrol.py uses to decide turn_left() vs turn_right(), so
    getting this backwards would make the robot turn the wrong way on
    every single correction."""
    angle = signed_angle_between((1, 0), (0, 1))
    assert angle > 0
    assert math.isclose(angle, math.pi / 2, abs_tol=1e-9)


def test_signed_angle_between_right_turn_is_negative():
    angle = signed_angle_between((1, 0), (0, -1))
    assert angle < 0
    assert math.isclose(angle, -math.pi / 2, abs_tol=1e-9)


def test_signed_angle_between_opposite_direction_is_pi():
    """180 degrees off -- sign is meaningless here (could report as +pi or
    -pi depending on floating point), but magnitude must be pi."""
    angle = signed_angle_between((1, 0), (-1, 0))
    assert math.isclose(abs(angle), math.pi, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# waypoint_xy / location_offset
# ---------------------------------------------------------------------------

WAYPOINTS = [
    {"id": "W1", "x": 15, "y": 20},
    {"id": "W2", "x": 50, "y": 12},
]


def test_waypoint_xy_found():
    assert waypoint_xy("W2", WAYPOINTS) == (50, 12)


def test_waypoint_xy_not_found_returns_none():
    assert waypoint_xy("W99", WAYPOINTS) is None


def test_location_offset_computes_correct_vector():
    result = location_offset("W1", {"x": 16.2, "y": 21.5}, WAYPOINTS)
    assert result == {"waypoint": "W1", "offset_x": 1.2, "offset_y": 1.5}


def test_location_offset_unknown_waypoint_returns_none():
    """A candidate raised while current_waypoint_id() somehow doesn't
    match any known WAYPOINTS entry must not crash the caller -- location
    logging is a nice-to-have, not something that should ever take down
    candidate-raising."""
    assert location_offset("W99", {"x": 0, "y": 0}, WAYPOINTS) is None


def test_location_offset_at_exact_waypoint_is_zero_offset():
    result = location_offset("W2", {"x": 50, "y": 12}, WAYPOINTS)
    assert result == {"waypoint": "W2", "offset_x": 0.0, "offset_y": 0.0}