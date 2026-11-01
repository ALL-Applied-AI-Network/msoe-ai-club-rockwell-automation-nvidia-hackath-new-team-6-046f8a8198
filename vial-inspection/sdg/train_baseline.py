#!/usr/bin/env python3
"""Train and evaluate a reproducible OpenCV HOG + SVM vial classifier baseline."""
import argparse
import csv
import json
from pathlib import Path
import random

import cv2
import numpy as np


def load_rows(dataset_root):
    with (Path(dataset_root)/"manifest.csv").open(newline="", encoding="utf-8") as handle:
        return [{**row, "_root": str(Path(dataset_root).resolve())}
                for row in csv.DictReader(handle)]


def hog_features(path, hog):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"could not decode image: {path}")
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image = cv2.equalizeHist(image)
    return hog.compute(image).reshape(-1)


def crop_path(row):
    """Resolve either the standard export's image field or pseudo export's path field."""
    return Path(row["_root"])/(row.get("path") or row["image"])


def confusion_matrix(actual, predicted, class_count):
    matrix = np.zeros((class_count, class_count), dtype=int)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    return matrix


def metrics_from_confusion(matrix):
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix)/total) if total else 0.0
    per_class = []
    for index in range(len(matrix)):
        tp = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp/predicted if predicted else 0.0
        recall = tp/support if support else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
        per_class.append({"support": support, "precision": precision,
                          "recall": recall, "f1": f1})
    return accuracy, per_class, sum(item["f1"] for item in per_class)/len(per_class)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path, nargs="+",
                        help="one or more compatible ImageFolder dataset roots")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=2.0)
    args = parser.parse_args()
    metadata = [json.loads((root/"dataset.json").read_text()) for root in args.dataset_root]
    classes = metadata[0]["classes"]
    if any(item.get("classes") != classes for item in metadata):
        parser.error("all dataset roots must use the same ordered classes")
    rows = [row for root in args.dataset_root for row in load_rows(root)]
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    args.output.mkdir(parents=True)

    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    partitions = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        random.Random(args.seed).shuffle(selected)
        features = np.asarray([hog_features(crop_path(row), hog)
                               for row in selected], dtype=np.float32)
        labels = np.asarray([int(row["class_id"]) for row in selected], dtype=np.int32)
        partitions[split] = (features, labels)
    if not all(len(partitions[split][1]) for split in partitions):
        parser.error("both train and val splits must contain samples")

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(args.c)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 2000, 1e-6))
    svm.train(partitions["train"][0], cv2.ml.ROW_SAMPLE, partitions["train"][1])
    _, prediction = svm.predict(partitions["val"][0])
    actual = partitions["val"][1]
    predicted = prediction.reshape(-1).astype(int)
    matrix = confusion_matrix(actual, predicted, len(classes))
    accuracy, per_class, macro_f1 = metrics_from_confusion(matrix)
    for name, values in zip(classes, per_class):
        values["class"] = name
    report = {"schema_version": "vial-baseline-1", "model": "opencv-hog-linear-svm",
              "seed": args.seed, "c": args.c,
              "train_samples": int(len(partitions["train"][1])),
              "val_samples": int(len(actual)), "classes": classes,
              "dataset_roots": [str(root.resolve()) for root in args.dataset_root],
              "accuracy": accuracy, "macro_f1": macro_f1,
              "confusion_matrix": matrix.tolist(), "per_class": per_class}
    svm.save(str(args.output/"model.yml"))
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"BASELINE VERIFIED accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} "
          f"train={report['train_samples']} val={report['val_samples']}")


if __name__ == "__main__":
    main()
