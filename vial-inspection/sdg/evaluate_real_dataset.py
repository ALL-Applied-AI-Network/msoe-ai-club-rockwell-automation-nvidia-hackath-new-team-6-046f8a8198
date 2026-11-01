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


def apply_two_stage_threshold(rows, threshold):
    for row in rows:
        if "defect_score" in row:
            row["predicted_class"] = (row["defect_class"]
                                      if row["defect_score"] > threshold else "pass")


def load_predictor(model_path, threshold_override=None):
    model_path = Path(model_path)
    if model_path.is_dir():
        metadata = json.loads((model_path/"model.json").read_text())
        if metadata.get("schema_version") != "vial-two-stage-1":
            raise ValueError("unsupported model directory")
        threshold = (metadata["defect_threshold"] if threshold_override is None
                     else threshold_override)
        return {"kind": "two_stage", "stage1": cv2.ml.SVM_load(str(model_path/"stage1.yml")),
                "stage2": cv2.ml.SVM_load(str(model_path/"stage2.yml")),
                "sign": metadata["positive_sign"], "threshold": threshold}
    if threshold_override is not None:
        raise ValueError("defect-threshold requires a two-stage model directory")
    return {"kind": "single", "model": cv2.ml.SVM_load(str(model_path))}


def predict(predictor, features):
    if predictor["kind"] == "single":
        _, values = predictor["model"].predict(features)
        return values.reshape(-1).astype(int), None, None
    _, raw = predictor["stage1"].predict(features, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    scores = raw.reshape(-1)*predictor["sign"]
    _, defects = predictor["stage2"].predict(features)
    defect_ids = defects.reshape(-1).astype(int)+1
    values = np.where(scores > predictor["threshold"], defect_ids, 0)
    return values, scores, defect_ids


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
    parser.add_argument("--defect-threshold", type=float)
    parser.add_argument("--threshold-profile", type=Path,
                        help="JSON profile containing a selected defect_threshold")
    parser.add_argument("--tune-threshold", action="store_true",
                        help="select a two-stage threshold on this split by macro agreement")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new file")
    paths = [path for path in sorted(args.images.glob("*.jpg"))
             if path.name not in set(args.exclude)]
    if not paths:
        parser.error("no non-excluded JPG images found")

    labeler = load_labeler(args.labeler)
    if args.defect_threshold is not None and args.threshold_profile is not None:
        parser.error("choose either defect-threshold or threshold-profile")
    threshold_override = args.defect_threshold
    if args.threshold_profile is not None:
        profile = json.loads(args.threshold_profile.read_text())
        if profile.get("schema_version") != "vial-two-stage-threshold-1":
            parser.error("unsupported threshold profile")
        threshold_override = float(profile["defect_threshold"])
    predictor = load_predictor(args.model, threshold_override)
    if args.tune_threshold and predictor["kind"] != "two_stage":
        parser.error("tune-threshold requires a two-stage model directory")
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
            predictions, scores, defect_ids = predict(predictor, features)
            image_rows = []
            for position, (slot, predicted_id) in enumerate(zip(selected, predictions)):
                row = {"image": path.name, "profile": result["alignment"]["profile"],
                       "slot": slot["id"], "reference_class": slot["class"],
                       "reference_confidence": slot["prediction_confidence"],
                       "reference_source": slot.get("prediction_source"),
                       "predicted_class": CLASSES[predicted_id]}
                if scores is not None:
                    row.update({"defect_score": float(scores[position]),
                                "defect_class": CLASSES[int(defect_ids[position])]})
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

    threshold_selection = None
    if args.tune_threshold:
        score_values = np.asarray([row["defect_score"] for row in rows])
        candidates = sorted(set(np.quantile(score_values, np.linspace(0.0, 1.0, 201)).tolist()+[0.0]))
        best = None
        for threshold in candidates:
            apply_two_stage_threshold(rows, threshold)
            result = summarize(rows)
            key = (result["macro_class_agreement"], result["agreement"], -abs(threshold))
            if best is None or key > best[0]:
                best = (key, threshold, result)
        _, selected_threshold, selected_summary = best
        apply_two_stage_threshold(rows, selected_threshold)
        threshold_selection = {"selected": selected_threshold,
                               "candidate_count": len(candidates),
                               "objective": "macro_class_agreement",
                               "macro_class_agreement": selected_summary["macro_class_agreement"],
                               "agreement": selected_summary["agreement"]}
        for item in per_image:
            image_rows = [row for row in rows if row["image"] == item["image"]]
            item["agreement"] = sum(row["reference_class"] == row["predicted_class"]
                                    for row in image_rows)/len(image_rows)

    summary = summarize(rows)
    report = {"schema_version": "vial-real-split-agreement-1",
              "warning": "Fixture mappings include machine-generated pseudo-labels; not certified ground truth.",
              "images_requested": len(paths), "images_evaluated": len(per_image),
              "areas": args.areas, "min_confidence": args.min_confidence,
              "model_kind": predictor["kind"],
              "defect_threshold": (threshold_selection["selected"] if threshold_selection
                                   else predictor.get("threshold")),
              "threshold_selection": threshold_selection,
              "threshold_profile": (str(args.threshold_profile.resolve())
                                    if args.threshold_profile else None),
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
