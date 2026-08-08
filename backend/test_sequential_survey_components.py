"""
test_sequential_survey_components.py -- real test coverage for the two
new modules that are safely import-testable without the hardware/GPU
stack: objects_registry.py (pure json + sort logic) and ocr_check.py's
cross_check() (pure comparison logic, no Tesseract binary required to
test the decision function itself).

HONEST GAP: real_feed.py's `_location_offset_for()` -- the waypoint+
offset vector math -- is NOT covered here. real_feed.py imports
`cyberwave` and loads YOLOE-26/OSNet at module level; importing it in a
test process risks failing on a machine without those installed/
configured, exactly the kind of environment-coupling this test file is
trying to avoid. `_location_offset_for()`'s logic is simple enough
(a single subtraction against a matched WAYPOINTS entry) that it's
covered indirectly by sequential_patrol.py's own `_location_offset()`
having the identical shape, but a direct unit test of real_feed's
version is a real, acknowledged gap -- not silently skipped, called out
here so it doesn't get mistaken for full coverage.

Run:
    python -m pytest backend/test_sequential_survey_components.py -v
"""

import json
import os
import tempfile

import pytest

from backend.objects_registry import load_objects, expected_number_for, RegisteredObject
from backend.vision.ocr_check import MarkerOCR, OCRResult


# ---------------------------------------------------------------------------
# objects_registry.py
# ---------------------------------------------------------------------------

def _write_config(tmp_path, objects):
    path = os.path.join(tmp_path, "objects_config.json")
    with open(path, "w") as f:
        json.dump({"objects": objects}, f)
    return path


def test_load_objects_sorts_by_number_not_file_order():
    """Locked decision: the sorted-by-number list IS the visit order --
    this must not depend on the order objects happen to appear in the
    JSON file, since a human editing the file shouldn't be able to
    silently reorder the survey by accident."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(tmp, [
            {"number": "03", "target_name": "object-03", "expected_waypoint": "W3", "twin_semantic_label": None},
            {"number": "01", "target_name": "object-01", "expected_waypoint": "W1", "twin_semantic_label": None},
            {"number": "02", "target_name": "object-02", "expected_waypoint": "W2", "twin_semantic_label": None},
        ])
        objects = load_objects(path)
        assert [o.number for o in objects] == ["01", "02", "03"]


def test_load_objects_missing_file_returns_empty_not_crash():
    """A missing config must degrade to 'nothing to survey', matching the
    same honesty posture as detector.py/reid.py's own missing-dependency
    handling -- never crash the caller over a missing optional file."""
    objects = load_objects("/nonexistent/path/objects_config.json")
    assert objects == []


def test_load_objects_preserves_none_twin_semantic_label():
    """Before the twin-sync chat's real schema is reconciled, labels are
    legitimately None -- load_objects must pass that through as None, not
    coerce it to an empty string or crash, since downstream code (Section
    5 Task 7) needs to distinguish 'not yet reconciled' from 'reconciled
    to empty'."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_config(tmp, [
            {"number": "01", "target_name": "object-01", "expected_waypoint": "W1", "twin_semantic_label": None},
        ])
        objects = load_objects(path)
        assert objects[0].twin_semantic_label is None


def test_expected_number_for_matches_by_target_name():
    objects = [
        RegisteredObject("01", "object-01", "W1", None),
        RegisteredObject("02", "object-02", "W2", "chair"),
    ]
    assert expected_number_for("object-02", objects) == "02"
    assert expected_number_for("unregistered-thing", objects) is None


# ---------------------------------------------------------------------------
# ocr_check.py -- cross_check() decision logic
# ---------------------------------------------------------------------------

def test_cross_check_agrees_returns_none():
    """OCR agreeing with OSNet's match is a non-event -- must not flag."""
    result = OCRResult(number="02", raw_text="02", confidence=90.0)
    assert MarkerOCR.cross_check(result, expected_number="02") is None


def test_cross_check_no_signal_returns_none_not_anomaly():
    """This is the specific rule the master doc locks in: 'no OCR signal'
    (nothing readable this frame) must NEVER be treated as an anomaly --
    only a genuine number MISMATCH is. A bad-angle frame is not evidence
    of anything wrong."""
    result = OCRResult(number=None, raw_text="", confidence=0.0)
    assert MarkerOCR.cross_check(result, expected_number="02") is None


def test_cross_check_mismatch_flags_anomaly():
    result = OCRResult(number="07", raw_text="07", confidence=88.0)
    anomaly = MarkerOCR.cross_check(result, expected_number="02")
    assert anomaly is not None
    assert "07" in anomaly and "02" in anomaly


def test_cross_check_zero_pads_expected_number():
    """expected_number may arrive as a bare int-like string ('2') from a
    caller that didn't zero-pad -- cross_check must normalize both sides
    the same way OCRResult.number already is (zfill(2) in read_number),
    or a correct match would falsely flag as a mismatch."""
    result = OCRResult(number="02", raw_text="02", confidence=90.0)
    assert MarkerOCR.cross_check(result, expected_number="2") is None


def test_marker_ocr_unavailable_degrades_gracefully():
    """If Tesseract's binary isn't installed, MarkerOCR must not crash at
    construction -- it degrades to unavailable, matching detector.py/
    reid.py's own pattern for optional heavy dependencies."""
    ocr = MarkerOCR()
    # Whichever way this environment resolves, the property access itself
    # must not raise -- that's the actual contract being tested here.
    assert ocr.available in (True, False)
    if not ocr.available:
        result = ocr.read_number(crop=None)
        assert result.number is None