#!/usr/bin/env python3
"""Create a balanced, image-split real crop dataset from unattended pseudo-labels."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

from PIL import Image, ImageOps


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_labeler(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("vial_participant_labeler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def balanced_select(candidates, seed, maximum):
    grouped = defaultdict(list)
    for item in candidates:
        grouped[(item["split"], item["class"])].append(item)
    selected = []
    for split in ("train", "val"):
        counts = [len(grouped[(split, name)]) for name in CLASSES]
        if not all(counts):
            raise ValueError(f"{split}: at least one class has no qualifying samples: {counts}")
        target = min(maximum, min(counts))
        for class_index, name in enumerate(CLASSES):
            values = grouped[(split, name)]
            random.Random(seed+class_index+(1000 if split == "val" else 0)).shuffle(values)
            selected.extend(values[:target])
    return sorted(selected, key=lambda item: (item["split"], item["class"],
                                               item["image"], item["slot"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--detections-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--max-per-class-split", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("val-fraction must be between 0 and 1")
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("min-confidence must be between 0 and 1")

    paths = sorted(args.images.glob("*.jpg"))
    excluded = set(args.exclude)
    paths = [path for path in paths if path.name not in excluded]
    if len(paths) < 2:
        parser.error("at least two non-excluded images are required")
    shuffled = list(paths)
    random.Random(args.seed).shuffle(shuffled)
    val_count = min(len(paths)-1, max(1, round(len(paths)*args.val_fraction)))
    val_names = {path.name for path in shuffled[:val_count]}

    labeler = load_labeler(args.labeler)
    candidates, failures = [], []
    for index, path in enumerate(paths, 1):
        try:
            results, _ = labeler.prepare([path], args.calibration, None, None,
                                         args.detections_root, True)
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append({"image": path.name, "error": str(error)})
            print(f"[{index}/{len(paths)}] SKIP {path.name}: {error}", flush=True)
            continue
        result = results[0]
        split = "val" if path.name in val_names else "train"
        for slot in result["slots"]:
            confidence = slot.get("prediction_confidence")
            if (not slot["id"].startswith("G") or slot.get("class") not in CLASSES
                    or confidence is None or confidence < args.min_confidence):
                continue
            candidates.append({"split": split, "class": slot["class"],
                               "image": path.name, "slot": slot["id"],
                               "bbox": slot["bbox"], "confidence": confidence,
                               "source": slot.get("prediction_source"),
                               "profile": result["alignment"]["profile"],
                               "source_sha256": result["source_sha256"]})
        print(f"[{index}/{len(paths)}] {path.name} candidates={len(candidates)}", flush=True)

    selected = balanced_select(candidates, args.seed, args.max_per_class_split)
    args.output.mkdir(parents=True)
    rows = []
    opened_name, opened = None, None
    try:
        for item in selected:
            if opened_name != item["image"]:
                if opened is not None:
                    opened.close()
                opened_name = item["image"]
                opened = ImageOps.exif_transpose(Image.open(args.images/opened_name)).convert("RGB")
            relative = Path(item["split"])/item["class"]/(
                f"{Path(item['image']).stem}__{item['slot']}.png")
            destination = args.output/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            opened.crop(item["bbox"]).save(destination)
            rows.append({**item, "class_id": CLASSES.index(item["class"]),
                         "crop_sha256": sha256(destination), "path": relative.as_posix()})
    finally:
        if opened is not None:
            opened.close()

    with (args.output/"manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {split: dict(sorted(Counter(row["class"] for row in rows
                                        if row["split"] == split).items()))
              for split in ("train", "val")}
    metadata = {"schema_version": "vial-imagefolder-1", "source": "real-pseudo-labels",
                "warning": "Machine-generated pseudo-labels; not human-certified ground truth.",
                "classes": list(CLASSES), "seed": args.seed,
                "min_confidence": args.min_confidence, "excluded_images": sorted(excluded),
                "source_images": len(paths), "aligned_images": len(paths)-len(failures),
                "sample_count": len(rows), "class_counts": counts, "failures": failures}
    (args.output/"dataset.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"REAL PSEUDO DATASET VERIFIED samples={len(rows)} images={len(paths)-len(failures)} "
          f"failures={len(failures)} train_per_class={next(iter(counts['train'].values()))} "
          f"val_per_class={next(iter(counts['val'].values()))}")


if __name__ == "__main__":
    main()
