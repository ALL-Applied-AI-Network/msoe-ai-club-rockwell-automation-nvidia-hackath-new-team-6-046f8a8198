#!/usr/bin/env python3
"""Calibrate/evaluate a tray-level defect threshold from per-position gate scores."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_real_dataset import extract_feature, load_predictor, predict


def load_labeler(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("vial_tray_calibration_labeler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict_metrics(records, threshold):
    classes = ("complete_order", "incomplete_order")
    matrix = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for record in records:
        predicted = ("incomplete_order" if record["max_defect_score"] > threshold
                     else "complete_order")
        matrix[record["actual"]][predicted] += 1
    recalls = {}
    for actual in classes:
        support = sum(matrix[actual].values())
        recalls[actual] = matrix[actual][actual]/support if support else None
    supported = [value for value in recalls.values() if value is not None]
    correct = sum(matrix[name][name] for name in classes)
    return {"accuracy": correct/len(records),
            "balanced_accuracy": sum(supported)/len(supported),
            "recall": recalls, "confusion_matrix": matrix}


def select_threshold(records):
    values = sorted({record["max_defect_score"] for record in records})
    candidates = [values[0]-1e-6, values[-1]+1e-6]
    candidates.extend((left+right)/2 for left, right in zip(values, values[1:]))
    best = None
    for threshold in candidates:
        metrics = verdict_metrics(records, threshold)
        key = (metrics["balanced_accuracy"], metrics["accuracy"], -abs(threshold))
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    return best[1], best[2], len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--threshold-profile", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.tune == (args.threshold_profile is not None):
        parser.error("choose exactly one of --tune or --threshold-profile")
    verdicts = {row["filename"]: row["verdict"]
                for row in csv.DictReader(args.verdicts.open(newline="", encoding="utf-8"))}
    paths = sorted(args.images.glob("*.jpg"))
    labeler = load_labeler(args.labeler)
    predictor = load_predictor(args.model, 0.0)
    if predictor["kind"] != "two_stage":
        parser.error("model must be a two-stage directory")
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    records, failures = [], []
    for index, path in enumerate(paths, 1):
        try:
            actual = verdicts[path.name]
            if actual not in {"complete_order", "incomplete_order"}:
                raise ValueError(f"unsupported verdict: {actual}")
            result = labeler.prepare([path], args.calibration)[0][0]
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            good = [slot for slot in result["slots"] if slot["id"].startswith("G")]
            features = np.asarray([extract_feature(image, slot["bbox"], hog)
                                   for slot in good], dtype=np.float32)
            _, scores, _ = predict(predictor, features)
            records.append({"image": path.name, "actual": actual,
                            "max_defect_score": float(np.max(scores))})
            print(f"[{index}/{len(paths)}] {path.name} actual={actual} "
                  f"max_score={records[-1]['max_defect_score']:.6f}", flush=True)
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append({"image": path.name, "error": str(error)})
    if not records:
        raise SystemExit("no evaluable tray verdicts")

    if args.tune:
        threshold, metrics, candidate_count = select_threshold(records)
        payload = {"schema_version": "vial-tray-threshold-1",
                   "defect_threshold": threshold, "selected_on": args.images.name,
                   "selection_objective": "balanced_accuracy",
                   "images": len(records), "candidate_count": candidate_count,
                   "metrics": metrics, "failures": failures,
                   "warning": "Threshold selected against supplied tray-level labels."}
    else:
        profile = json.loads(args.threshold_profile.read_text())
        if profile.get("schema_version") != "vial-tray-threshold-1":
            parser.error("unsupported tray threshold profile")
        threshold = float(profile["defect_threshold"])
        payload = {"schema_version": "vial-tray-evaluation-1",
                   "defect_threshold": threshold, "evaluated_on": args.images.name,
                   "images": len(records), "metrics": verdict_metrics(records, threshold),
                   "failures": failures, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2)+"\n")
    metrics = payload["metrics"]
    print(f"TRAY VERDICT VERIFIED images={len(records)} threshold={threshold:.6f} "
          f"accuracy={metrics['accuracy']:.4f} balanced={metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
