"""
objects_registry.py -- loads and reconciles objects_config.json.

Deliberately a thin, standalone module rather than folded into
real_feed.py -- this is the piece most likely to need reshaping once
the twin-sync chat's real output format is known (master doc Section
6), so keeping it isolated means that reconciliation touches one small
file, not real_feed.py's core candidate-raising logic.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger("quarry.objects_registry")

DEFAULT_PATH = os.path.join(os.path.dirname(__file__), "objects_config.json")


@dataclass
class RegisteredObject:
    number: str               # zero-padded sequence position, e.g. "01"
    target_name: str          # must match the name passed to state.register_target()
    expected_waypoint: str    # must be one of real_feed.WAYPOINTS[*]["id"]
    twin_semantic_label: Optional[str]  # owned by the twin-sync chat, may be None until reconciled


def load_objects(path: str = DEFAULT_PATH) -> List[RegisteredObject]:
    """Returns objects sorted by `number` -- this sorted list IS the
    sequential visit order the nav controller drives, per the master
    doc's locked decision ("Target sequence = ordered list of registered
    object numbers, not a live path-planning problem")."""
    if not os.path.exists(path):
        logger.warning(
            "objects_config.json not found at %s -- sequential controller "
            "has nothing to survey. Create it (see docs/objects_config.json "
            "for the schema) before running sequential_patrol.py.", path,
        )
        return []

    with open(path, "r") as f:
        raw = json.load(f)

    objects = [
        RegisteredObject(
            number=o["number"],
            target_name=o["target_name"],
            expected_waypoint=o["expected_waypoint"],
            twin_semantic_label=o.get("twin_semantic_label"),
        )
        for o in raw.get("objects", [])
    ]
    objects.sort(key=lambda o: o.number)

    unlabeled = [o.number for o in objects if o.twin_semantic_label is None]
    if unlabeled:
        logger.info(
            "objects_config.json: %d object(s) have no twin_semantic_label yet "
            "(numbers: %s) -- fine for standalone nav/OCR work, but reconcile "
            "against the twin-sync chat's real schema before Section 5 Task 7 "
            "(dual real-feed + twin-view) needs it.", len(unlabeled), unlabeled,
        )

    return objects


def expected_number_for(target_name: str, objects: List[RegisteredObject]) -> Optional[str]:
    """Given an OSNet-matched target name, returns its expected marker
    number for OCR cross-checking, or None if it's not in the registry
    (e.g. a generic non-sequential detection)."""
    for o in objects:
        if o.target_name == target_name:
            return o.number
    return None