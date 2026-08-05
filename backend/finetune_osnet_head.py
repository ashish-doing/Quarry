"""
finetune_osnet_head.py -- fine-tunes OSNet's classification head on
QUARRY's own confirmed/disputed vote data. Does NOT touch the backbone
(realistic for the RTX 4050's 6GB budget alongside YOLOE-26, per the
master doc's re-id approach section).

READ THIS BEFORE RUNNING: this script only makes sense once real hunt
sessions have generated training_data/ via site_agent.py's
_save_training_example(). It refuses to run against synthetic/empty
data and reports exactly why, rather than producing a number that looks
like a result.

WHAT THIS ACTUALLY FINE-TUNES: only REGISTERED TARGET identities, not
generic YOLOE object classes. training_data/<confirmed|disputed>/<label>/
mixes two different things depending on what real_feed.py decided at
detection time -- a folder named after a registered target (e.g.
"intruder-01") means the re-id matcher succeeded; a folder named after a
generic class (e.g. "backpack", "person") means it never matched a
registered target at all. Only the first kind is a genuine re-id signal
-- fine-tuning on "confirmed backpack vs disputed backpack" would be
training an object-confidence classifier, not a re-id head, and would
silently corrupt what "fine-tuned OSNet" means in the pitch. This script
cross-checks label folder names against real_feed.state's registered
target list at runtime (or, if that's not importable standalone, against
a --registered-targets file you provide) and skips anything that doesn't
match, logging exactly what was skipped and why.

Usage:
    python backend/finetune_osnet_head.py --check-only
        # just reports data volume per registered-target class, no training

    python backend/finetune_osnet_head.py --registered-targets targets.txt
        # targets.txt: one registered target name per line, matching
        # whatever names were used in register_target() calls

Outputs training_data/finetune_report.json with honest before/after
numbers -- including sample counts, so a small improvement on 12 crops
isn't presented with the same confidence as one on 500.
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("quarry.finetune")

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "training_data")
REPORT_PATH = os.path.join(TRAINING_DATA_DIR, "finetune_report.json")

MIN_SAMPLES_PER_CLASS = 8    # below this, a train/eval split is nearly meaningless -- refuse, don't fudge
MIN_CLASSES = 2              # a classifier head needs at least 2 identities to distinguish
EVAL_FRACTION = 0.3
RANDOM_SEED = 42


def _load_registered_target_names(path: str = None) -> List[str]:
    """Real registered-target names to fine-tune against. Prefer importing
    real_feed.state directly (the actual source of truth, same object
    site_agent.py talks to) -- fall back to a plain text file only if
    that import isn't possible from wherever this script is run."""
    if path:
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    try:
        from backend import real_feed
        return [t["name"] for t in real_feed.state.registered_targets]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not import backend.real_feed to read registered targets live (%s). "
            "Pass --registered-targets a_file.txt instead, or run this from the repo "
            "root so `from backend import real_feed` resolves.", exc,
        )
        return []


def _scan_training_data(registered_names: List[str]) -> Dict[str, Dict[str, List[str]]]:
    """Returns {class_name: {"confirmed": [paths], "disputed": [paths]}},
    restricted to class_names that are actual registered target names --
    everything else is logged and skipped, per the module docstring."""
    result: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: {"confirmed": [], "disputed": []})
    skipped_generic = []

    for outcome in ("confirmed", "disputed"):
        outcome_dir = os.path.join(TRAINING_DATA_DIR, outcome)
        if not os.path.isdir(outcome_dir):
            continue
        for label in os.listdir(outcome_dir):
            label_dir = os.path.join(outcome_dir, label)
            if not os.path.isdir(label_dir):
                continue
            if label not in registered_names:
                skipped_generic.append((outcome, label))
                continue
            files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".jpg")]
            result[label][outcome].extend(files)

    if skipped_generic:
        logger.info(
            "Skipped %d generic-object folder(s) (not registered-target identities, "
            "not a re-id signal): %s",
            len(skipped_generic), sorted(set(f"{o}/{l}" for o, l in skipped_generic)),
        )
    return dict(result)


def _report_volume(data: Dict[str, Dict[str, List[str]]]):
    print("\n--- Registered-target training data volume ---")
    if not data:
        print("  (none found)")
        return
    for label, d in sorted(data.items()):
        n_conf, n_disp = len(d["confirmed"]), len(d["disputed"])
        status = "OK" if n_conf >= MIN_SAMPLES_PER_CLASS else f"NEEDS {MIN_SAMPLES_PER_CLASS - n_conf} more confirmed"
        print(f"  {label}: {n_conf} confirmed, {n_disp} disputed  [{status}]")
    print()


def _eligible_classes(data: Dict[str, Dict[str, List[str]]]) -> List[str]:
    return [label for label, d in data.items() if len(d["confirmed"]) >= MIN_SAMPLES_PER_CLASS]


def _split(files: List[str], eval_fraction: float) -> Tuple[List[str], List[str]]:
    rng = random.Random(RANDOM_SEED)
    shuffled = files[:]
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_fraction))
    return shuffled[n_eval:], shuffled[:n_eval]


def _load_image(path: str) -> np.ndarray:
    import cv2
    bgr = cv2.imread(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _extract_embeddings(extractor, paths: List[str]) -> np.ndarray:
    images = [_load_image(p) for p in paths]
    feats = extractor(images)
    vecs = np.stack([f.cpu().numpy() if hasattr(f, "cpu") else np.asarray(f) for f in feats])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def _baseline_accuracy(train_embeds: Dict[str, np.ndarray], eval_embeds: Dict[str, np.ndarray]) -> float:
    """Same logic as reid.py's TargetMatcher.best_match: nearest neighbor
    by cosine similarity across each class's raw (un-fine-tuned) gallery.
    This IS the current production behavior -- the honest "before" number."""
    correct, total = 0, 0
    for true_label, embeds in eval_embeds.items():
        for vec in embeds:
            best_label, best_score = None, -1.0
            for cand_label, gallery in train_embeds.items():
                score = float(np.max(gallery @ vec))
                if score > best_score:
                    best_label, best_score = cand_label, score
            correct += int(best_label == true_label)
            total += 1
    return correct / total if total else 0.0


def _train_head(train_embeds: Dict[str, np.ndarray], input_dim: int):
    """A small linear classification head on top of frozen OSNet
    embeddings -- NOT fine-tuning the backbone (matches the master doc's
    VRAM-budget reasoning). Plain logistic regression is deliberately
    used instead of a hand-rolled PyTorch loop: with class counts this
    small (single/low-double digits per class, gated by
    MIN_SAMPLES_PER_CLASS), a linear head trained with proper
    regularization is both more honest and less likely to overfit than
    a deeper head would be."""
    from sklearn.linear_model import LogisticRegression

    labels = sorted(train_embeds.keys())
    X, y = [], []
    for label in labels:
        for vec in train_embeds[label]:
            X.append(vec)
            y.append(label)
    X = np.stack(X)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def _head_accuracy(clf, eval_embeds: Dict[str, np.ndarray]) -> float:
    correct, total = 0, 0
    for true_label, embeds in eval_embeds.items():
        for vec in embeds:
            pred = clf.predict(vec.reshape(1, -1))[0]
            correct += int(pred == true_label)
            total += 1
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true",
                         help="Only report data volume per class, don't train.")
    parser.add_argument("--registered-targets", default=None,
                         help="Text file, one registered target name per line. "
                              "Omit to auto-import from backend.real_feed.state.")
    args = parser.parse_args()

    registered_names = _load_registered_target_names(args.registered_targets)
    if not registered_names:
        print("No registered target names available (empty list from real_feed.state, "
              "and no --registered-targets file given). Nothing to fine-tune against -- "
              "this is expected if no targets have been registered in a real session yet.")
        return

    data = _scan_training_data(registered_names)
    _report_volume(data)

    eligible = _eligible_classes(data)
    if len(eligible) < MIN_CLASSES:
        print(f"Only {len(eligible)} class(es) have >= {MIN_SAMPLES_PER_CLASS} confirmed "
              f"samples (need >= {MIN_CLASSES} to fine-tune a classifier at all). "
              f"This is the expected state before real hunt sessions have run -- "
              f"not an error. Run more sessions, then re-check.")
        return

    if args.check_only:
        print(f"{len(eligible)} class(es) eligible for fine-tuning: {eligible}")
        return

    print(f"Loading OSNet backbone (frozen, feature extraction only)...")
    from torchreid.reid.utils import FeatureExtractor
    extractor = FeatureExtractor(model_name="osnet_x0_25", model_path="", device="cuda")

    train_embeds, eval_embeds = {}, {}
    for label in eligible:
        train_files, eval_files = _split(data[label]["confirmed"], EVAL_FRACTION)
        if not eval_files:
            print(f"  '{label}': too few samples to hold out an eval split, skipping this class")
            continue
        train_embeds[label] = _extract_embeddings(extractor, train_files)
        eval_embeds[label] = _extract_embeddings(extractor, eval_files)
        print(f"  '{label}': {len(train_files)} train, {len(eval_files)} eval")

    if len(train_embeds) < MIN_CLASSES:
        print("Not enough classes survived the train/eval split to fine-tune. "
              "Need more confirmed samples per registered target.")
        return

    print("\nComputing baseline (current production nearest-neighbor matching)...")
    baseline_acc = _baseline_accuracy(train_embeds, eval_embeds)

    print("Training linear head on frozen embeddings...")
    clf = _train_head(train_embeds, input_dim=next(iter(train_embeds.values())).shape[1])

    print("Evaluating fine-tuned head...")
    finetuned_acc = _head_accuracy(clf, eval_embeds)

    n_eval_total = sum(len(v) for v in eval_embeds.values())
    report = {
        "classes": sorted(train_embeds.keys()),
        "n_train_samples": sum(len(v) for v in train_embeds.values()),
        "n_eval_samples": n_eval_total,
        "baseline_nearest_neighbor_accuracy": round(baseline_acc, 4),
        "finetuned_head_accuracy": round(finetuned_acc, 4),
        "delta": round(finetuned_acc - baseline_acc, 4),
        "note": (
            f"Evaluated on only {n_eval_total} held-out samples -- treat this delta as "
            f"a directional signal, not a statistically robust claim, until real hunt "
            f"session volume grows. Report both numbers together, not the delta alone."
        ),
    }
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Results (n_eval={n_eval_total}) ---")
    print(f"  Baseline (current production):  {baseline_acc:.1%}")
    print(f"  Fine-tuned head:                {finetuned_acc:.1%}")
    print(f"  Delta:                          {finetuned_acc - baseline_acc:+.1%}")
    print(f"\nSaved full report to {REPORT_PATH}")
    if n_eval_total < 20:
        print(f"\nCAUTION: only {n_eval_total} eval samples. Report this number alongside "
              f"the accuracy in the pitch -- an unqualified percentage here would overstate "
              f"confidence the data doesn't support.")


if __name__ == "__main__":
    main()