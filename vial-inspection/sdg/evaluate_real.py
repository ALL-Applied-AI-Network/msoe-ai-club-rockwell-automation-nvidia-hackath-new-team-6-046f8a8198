#!/usr/bin/env python3
"""Measure agreement between a synthetic-trained baseline and real pseudo-labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}


def extract_feature(image, bbox, hog):
    left, top, right, bottom = bbox
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError(f"empty crop for bbox {bbox}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    return hog.compute(gray).reshape(-1)


def agreement_summary(rows):
    agreement = sum(row["reference_class"] == row["predicted_class"] for row in rows)
    by_reference = {}
    for name in CLASSES:
        selected = [row for row in rows if row["reference_class"] == name]
        matches = sum(row["predicted_class"] == name for row in selected)
        by_reference[name] = {"count": len(selected),
                              "agreement": matches/len(selected) if selected else None}
    return agreement/len(rows) if rows else 0.0, by_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--areas", choices=("good", "all"), default="good")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    model = cv2.ml.SVM_load(str(args.model))
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    rows, images = [], []
    annotation_paths = sorted(args.annotations.glob("*.predicted.json"))
    if not annotation_paths:
        parser.error("no *.predicted.json annotations found")

    for annotation_path in annotation_paths:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        result = payload["results"][0]
        image_path = args.images/result["filename"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not decode source image: {image_path}")
        selected = [slot for slot in result["slots"]
                    if args.areas == "all" or slot["id"].startswith("G")]
        features = np.asarray([extract_feature(image, slot["bbox"], hog)
                               for slot in selected], dtype=np.float32)
        _, predictions = model.predict(features)
        image_rows = []
        for slot, predicted_id in zip(selected, predictions.reshape(-1).astype(int)):
            row = {"image": result["filename"], "profile": result["alignment"]["profile"],
                   "slot": slot["id"], "reference_class": slot["class"],
                   "reference_confidence": slot.get("prediction_confidence"),
                   "reference_source": slot.get("prediction_source"),
                   "predicted_class": CLASSES[predicted_id]}
            rows.append(row)
            image_rows.append(row)
        score, _ = agreement_summary(image_rows)
        images.append({"image": result["filename"],
                       "profile": result["alignment"]["profile"],
                       "samples": len(image_rows), "agreement": score,
                       "reference_counts": dict(sorted(Counter(
                           row["reference_class"] for row in image_rows).items())),
                       "prediction_counts": dict(sorted(Counter(
                           row["predicted_class"] for row in image_rows).items()))})

    agreement, by_reference = agreement_summary(rows)
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    for row in rows:
        matrix[CLASS_TO_ID[row["reference_class"]]][CLASS_TO_ID[row["predicted_class"]]] += 1
    report = {"schema_version": "vial-real-agreement-1",
              "warning": "Reference annotations are machine-generated pseudo-labels, not ground truth.",
              "areas": args.areas, "samples": len(rows), "agreement": agreement,
              "classes": list(CLASSES), "agreement_matrix": matrix,
              "by_reference_class": by_reference, "images": images,
              "disagreements": [row for row in rows
                                if row["reference_class"] != row["predicted_class"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"REAL AGREEMENT VERIFIED samples={len(rows)} agreement={agreement:.4f} "
          f"disagreements={len(report['disagreements'])}")


if __name__ == "__main__":
    main()
