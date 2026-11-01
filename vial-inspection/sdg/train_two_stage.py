#!/usr/bin/env python3
"""Train a pass/defect gate followed by a four-class defect SVM."""
import argparse
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")
DEFECT_CLASSES = CLASSES[1:]


def load_rows(root):
    with (Path(root)/"manifest.csv").open(newline="", encoding="utf-8") as handle:
        return [{**row, "_root": str(Path(root).resolve())} for row in csv.DictReader(handle)]


def crop_path(row):
    return Path(row["_root"])/(row.get("path") or row["image"])


def feature(path, hog):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    return hog.compute(cv2.equalizeHist(image)).reshape(-1)


def make_svm(c, class_weights):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(c)
    svm.setClassWeights(np.asarray(class_weights, dtype=np.float32))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 2000, 1e-6))
    return svm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--pass-weight", type=float, default=4.0,
                        help="stage-one cost for misclassifying pass")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if args.pass_weight <= 0:
        parser.error("pass-weight must be positive")
    metadata = [json.loads((root/"dataset.json").read_text()) for root in args.dataset_root]
    if any(item.get("classes") != list(CLASSES) for item in metadata):
        parser.error("all datasets must use the canonical ordered classes")
    rows = [row for root in args.dataset_root for row in load_rows(root)]
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    partitions = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        random.Random(args.seed).shuffle(selected)
        features = np.asarray([feature(crop_path(row), hog) for row in selected], dtype=np.float32)
        labels = np.asarray([int(row["class_id"]) for row in selected], dtype=np.int32)
        partitions[split] = (features, labels)

    train_x, train_y = partitions["train"]
    stage1_y = (train_y != 0).astype(np.int32)
    stage1 = make_svm(args.c, [args.pass_weight, 1.0])
    stage1.train(train_x, cv2.ml.ROW_SAMPLE, stage1_y)
    defect_mask = train_y != 0
    stage2 = make_svm(args.c, [1.0]*4)
    stage2.train(train_x[defect_mask], cv2.ml.ROW_SAMPLE, train_y[defect_mask]-1)

    _, train_raw = stage1.predict(train_x, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    raw = train_raw.reshape(-1)
    positive_sign = 1.0 if raw[stage1_y == 1].mean() > raw[stage1_y == 0].mean() else -1.0
    val_x, val_y = partitions["val"]
    _, val_raw = stage1.predict(val_x, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    defect_score = val_raw.reshape(-1)*positive_sign
    _, defect_prediction = stage2.predict(val_x)
    predicted = np.where(defect_score > 0.0,
                         defect_prediction.reshape(-1).astype(int)+1, 0)
    accuracy = float((predicted == val_y).mean())

    args.output.mkdir(parents=True)
    stage1.save(str(args.output/"stage1.yml"))
    stage2.save(str(args.output/"stage2.yml"))
    report = {"schema_version": "vial-two-stage-1", "classes": list(CLASSES),
              "defect_classes": list(DEFECT_CLASSES), "seed": args.seed,
              "c": args.c, "pass_weight": args.pass_weight,
              "positive_sign": positive_sign, "defect_threshold": 0.0,
              "train_samples": int(len(train_y)), "val_samples": int(len(val_y)),
              "default_threshold_val_accuracy": accuracy,
              "dataset_roots": [str(root.resolve()) for root in args.dataset_root]}
    (args.output/"model.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"TWO STAGE VERIFIED accuracy={accuracy:.4f} train={len(train_y)} "
          f"val={len(val_y)} sign={positive_sign:+.0f}")


if __name__ == "__main__":
    main()
