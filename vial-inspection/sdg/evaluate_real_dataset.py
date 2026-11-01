#!/usr/bin/env python3
"""Evaluate a trained crop model across an untouched real dataset split."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def load_labeler(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("vial_eval_labeler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_feature(image, bbox, hog):
    left, top, right, bottom = bbox
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"empty crop for bbox {bbox}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return hog.compute(gray).reshape(-1)


def summarize(rows):
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    per_class = {}
    for row in rows:
        matrix[CLASS_TO_ID[row["reference_class"]]][CLASS_TO_ID[row["predicted_class"]]] += 1
    for name in CLASSES:
        selected = [row for row in rows if row["reference_class"] == name]
        matches = sum(row["predicted_class"] == name for row in selected)
        per_class[name] = {"count": len(selected),
                           "agreement": matches/len(selected) if selected else None}
    supported = [item["agreement"] for item in per_class.values()
                 if item["agreement"] is not None]
    return {"agreement": sum(row["reference_class"] == row["predicted_class"] for row in rows)/len(rows),
            "macro_class_agreement": sum(supported)/len(supported),
            "per_class": per_class, "agreement_matrix": matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--detections-root", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--areas", choices=("good", "all"), default="good")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    paths = [path for path in sorted(args.images.glob("*.jpg"))
             if path.name not in set(args.exclude)]
    if not paths:
        parser.error("no non-excluded JPG images found")

    labeler = load_labeler(args.labeler)
    model = cv2.ml.SVM_load(str(args.model))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    rows, failures, per_image = [], [], []
    for index, path in enumerate(paths, 1):
        try:
            results, _ = labeler.prepare([path], args.calibration, None, None,
                                         args.detections_root, True)
            result = results[0]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            selected = [slot for slot in result["slots"]
                        if (args.areas == "all" or slot["id"].startswith("G"))
                        and slot.get("prediction_confidence", -1) >= args.min_confidence]
            if not selected:
                raise ValueError("no positions met the confidence threshold")
            features = np.asarray([extract_feature(image, slot["bbox"], hog)
                                   for slot in selected], dtype=np.float32)
            _, predictions = model.predict(features)
            image_rows = []
            for slot, predicted_id in zip(selected, predictions.reshape(-1).astype(int)):
                row = {"image": path.name, "profile": result["alignment"]["profile"],
                       "slot": slot["id"], "reference_class": slot["class"],
                       "reference_confidence": slot["prediction_confidence"],
                       "reference_source": slot.get("prediction_source"),
                       "predicted_class": CLASSES[predicted_id]}
                rows.append(row)
                image_rows.append(row)
            image_matches = sum(item["reference_class"] == item["predicted_class"]
                                for item in image_rows)
            per_image.append({"image": path.name, "profile": result["alignment"]["profile"],
                              "samples": len(image_rows),
                              "agreement": image_matches/len(image_rows)})
            print(f"[{index}/{len(paths)}] {path.name} samples={len(image_rows)}", flush=True)
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append({"image": path.name, "error": str(error)})
            print(f"[{index}/{len(paths)}] SKIP {path.name}: {error}", flush=True)
    if not rows:
        raise SystemExit("no evaluable positions")

    summary = summarize(rows)
    report = {"schema_version": "vial-real-split-agreement-1",
              "warning": "Fixture mappings include machine-generated pseudo-labels; not certified ground truth.",
              "images_requested": len(paths), "images_evaluated": len(per_image),
              "areas": args.areas, "min_confidence": args.min_confidence,
              "samples": len(rows), **summary,
              "reference_counts": dict(sorted(Counter(
                  row["reference_class"] for row in rows).items())),
              "prediction_counts": dict(sorted(Counter(
                  row["predicted_class"] for row in rows).items())),
              "failures": failures, "images": per_image,
              "disagreements": [row for row in rows
                                if row["reference_class"] != row["predicted_class"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(f"REAL SPLIT VERIFIED images={len(per_image)} samples={len(rows)} "
          f"agreement={summary['agreement']:.4f} "
          f"macro={summary['macro_class_agreement']:.4f} failures={len(failures)}")


if __name__ == "__main__":
    main()
