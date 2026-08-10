"""
register_mission_objects.py -- offline, one-time registration of the 8
numbered mission objects. Run this BEFORE starting inner_agent.py --
there is no live in-session "+TARGET" UI anymore (see main.py's module
docstring for what was removed and why).

WHAT THIS DOES: for each object in objects_config.json, loads 3-5 real
photos from a local folder and builds an OSNet embedding gallery
(TargetMatcher.register()), then saves that gallery to disk
(target_gallery.pkl). inner_agent.py's RealRobotState automatically
loads that file at startup -- see reid.py's TargetMatcher.load_gallery()
and real_feed.py's RealRobotState.__init__.

WHY THIS IS A SEPARATE SCRIPT, NOT PART OF inner_agent.py ITSELF:
registering good multi-angle embeddings is a one-time, unhurried task --
you want to take your time getting clean photos of each object before
the mission starts, not do it under time pressure while the robot is
already trying to drive. Running it standalone also means you can
re-register a single object (better lighting, a cleaner angle) without
touching the mission-running process at all.

FOLDER LAYOUT EXPECTED:
    mission_objects/
      object-01/  *.jpg | *.png   (3-5 photos, different angles)
      object-02/  ...
      ...
      object-08/  ...

The folder name must exactly match objects_config.json's target_name
for that object -- this script reads the expected names FROM
objects_config.json itself (via objects_registry.py), so it will tell
you plainly if a folder is missing rather than silently skipping it.

Usage:
    python -m backend.register_mission_objects
    python -m backend.register_mission_objects --photos-dir /path/to/mission_objects
"""

import argparse
import glob
import os

import cv2

from backend.objects_registry import load_objects
from backend.vision.reid import TargetMatcher

DEFAULT_PHOTOS_DIR = os.path.join(os.path.dirname(__file__), "..", "mission_objects")
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _load_photos_for(target_name: str, photos_dir: str):
    folder = os.path.join(photos_dir, target_name)
    if not os.path.isdir(folder):
        return None  # distinguished from "folder exists but empty" below

    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith(VALID_EXTENSIONS)
    )
    photos = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"    [skip] couldn't read {os.path.basename(path)} -- corrupt or unsupported format")
            continue
        photos.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return photos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos-dir", default=DEFAULT_PHOTOS_DIR,
                         help=f"Folder containing one subfolder per object (default: {DEFAULT_PHOTOS_DIR})")
    args = parser.parse_args()

    objects = load_objects()
    if not objects:
        print("No objects found in objects_config.json -- nothing to register. "
              "Check the file exists and has an 'objects' list.")
        return

    print(f"Registering {len(objects)} mission object(s) from {args.photos_dir}\n")

    matcher = TargetMatcher()
    if not matcher.available:
        print("[FATAL] OSNet backend unavailable (check `pip install torchreid` and CUDA) -- "
              "cannot build embeddings. Registration cannot proceed without this; a re-id "
              "gallery with zero embeddings is not a usable fallback, it just means every "
              "candidate will come back unregistered during the mission.")
        return

    registered_count = 0
    problems = []

    for obj in objects:
        desc = f" ({obj.description})" if obj.description else ""
        print(f"  object-{obj.number} -> '{obj.target_name}'{desc}")
        photos = _load_photos_for(obj.target_name, args.photos_dir)

        if photos is None:
            msg = f"    [MISSING] no folder at {os.path.join(args.photos_dir, obj.target_name)}"
            print(msg)
            problems.append(f"object-{obj.number} ('{obj.target_name}'): no photo folder found")
            continue

        if not photos:
            msg = f"    [EMPTY] folder exists but no readable photos in it"
            print(msg)
            problems.append(f"object-{obj.number} ('{obj.target_name}'): folder empty or unreadable")
            continue

        if len(photos) < 3:
            print(f"    [WARN] only {len(photos)} photo(s) -- 3-5 angles recommended for "
                  f"real re-id robustness, proceeding anyway with what's here")

        matcher.register(obj.target_name, photos)
        print(f"    registered with {len(photos)} photo(s)")
        registered_count += 1

    print()
    if registered_count == 0:
        print("[FATAL] Nothing was registered -- gallery NOT saved. Fix the folder layout and re-run.")
        return

    saved = matcher.save_gallery()
    if saved:
        print(f"Saved gallery: {registered_count}/{len(objects)} object(s) registered.")
    else:
        print("[FATAL] Registration succeeded in-memory but save_gallery() failed -- "
              "check disk permissions/space. Nothing persisted; inner_agent.py will "
              "start with an empty gallery until this is fixed.")

    if problems:
        print(f"\n{len(problems)} object(s) NOT registered:")
        for p in problems:
            print(f"  - {p}")
        print("\nThese will raise as unregistered generic detections during the mission "
              "until fixed and this script is re-run.")


if __name__ == "__main__":
    main()