#!/usr/bin/env python3
"""Run leakage-free 60-position vial inspection and emit auditable tray verdicts."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_real_dataset import CLASSES, extract_feature, load_predictor, predict
from train_tray_classifier import VERDICTS, tray_feature_vector


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tray_verdict(groups):
    complete = groups["good"]["findings"]["complete_order"]
    if complete is True:
        return "complete_order"
    if complete is False:
        return "incomplete_order"
    return "unresolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--files", required=True, nargs="+")
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path,
                        help="two-stage model directory")
    parser.add_argument("--threshold-profile", required=True, type=Path)
    parser.add_argument("--tray-model", type=Path,
                        help="optional trained tray classifier directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--good-expected-through", type=int, default=30)
    parser.add_argument("--scrap-expected-through", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if any(Path(name).name != name for name in args.files):
        parser.error("files must be image basenames")
    profile = json.loads(args.threshold_profile.read_text())
    if profile.get("schema_version") not in {"vial-two-stage-threshold-1",
                                              "vial-tray-threshold-1"}:
        parser.error("unsupported threshold profile")
    threshold = float(profile["defect_threshold"])
    predictor = load_predictor(args.model, threshold)
    if predictor["kind"] != "two_stage":
        parser.error("model must be a two-stage model directory")
    tray_model = tray_scaler = tray_metadata = None
    if args.tray_model is not None:
        tray_metadata = json.loads((args.tray_model/"metrics.json").read_text())
        if tray_metadata.get("schema_version") != "vial-tray-classifier-1":
            parser.error("unsupported tray classifier")
        tray_model = cv2.ml.SVM_load(str(args.tray_model/"model.yml"))
        with np.load(args.tray_model/"scaler.npz") as values:
            tray_scaler = (values["mean"], values["std"])

    labeler = load_module("vial_inference_labeler", args.labeler)
    prediction_module = __import__("prediction")
    rules = __import__("rules")
    paths = [args.images/name for name in args.files]
    for path in paths:
        if not path.is_file():
            parser.error(f"image does not exist: {path}")
    args.output.mkdir(parents=True)
    empty_yolo = args.output/"empty_detections.txt"
    empty_yolo.write_text("")
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    results, position_rows, summary_rows = [], [], []

    try:
        for path in paths:
            prepared, _ = labeler.prepare([path], args.calibration)
            result = prepared[0]
            pil_image = Image.open(path).convert("RGB")
            # Empty detections intentionally activate appearance/fill fallback only.
            prediction_module.predict(result, pil_image, empty_yolo, auto_confirm=False)
            pil_image.close()
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            good = [slot for slot in result["slots"] if slot["id"].startswith("G")]
            features = np.asarray([extract_feature(image, slot["bbox"], hog)
                                   for slot in good], dtype=np.float32)
            predicted, scores, defect_ids = predict(predictor, features)
            for index, (slot, class_id) in enumerate(zip(good, predicted)):
                label = CLASSES[int(class_id)]
                slot.update({"class": label, "presence": label != "no_vial",
                             "suggested_class": label, "review_status": "predicted",
                             "prediction_source": "two_stage_hog_svm",
                             "prediction_confidence": None,
                             "defect_score": round(float(scores[index]), 8),
                             "defect_threshold": threshold,
                             "defect_candidate": CLASSES[int(defect_ids[index])],
                             "reason": "model:two_stage_hog_svm; score is not a probability"})
            for slot in result["slots"]:
                if slot["id"].startswith("S"):
                    slot["review_status"] = "predicted"
            result["groups"] = {
                "good": rules.summarize("good", good, args.good_expected_through),
                "scrap": rules.summarize("scrap",
                                          [slot for slot in result["slots"]
                                           if slot["id"].startswith("S")],
                                          args.scrap_expected_through),
            }
            result["rule_tray_verdict"] = tray_verdict(result["groups"])
            if tray_model is not None:
                tray_features = tray_feature_vector(scores, defect_ids,
                                                    result["alignment"]["profile"],
                                                    tray_metadata["feature_set"])
                mean, std = tray_scaler
                _, tray_prediction = tray_model.predict(
                    ((tray_features-mean)/std).reshape(1, -1).astype(np.float32))
                result["tray_verdict"] = VERDICTS[int(tray_prediction[0, 0])]
                result["tray_model"] = {
                    "feature_set": tray_metadata["feature_set"],
                    "model_sha256": sha256(args.tray_model/"model.yml"),
                    "scaler_sha256": sha256(args.tray_model/"scaler.npz"),
                }
            else:
                result["tray_verdict"] = result["rule_tray_verdict"]
            result["backend"] = "two_stage_hog_svm_plus_scrap_appearance"
            result["model"] = {"directory": str(args.model.resolve()),
                               "stage1_sha256": sha256(args.model/"stage1.yml"),
                               "stage2_sha256": sha256(args.model/"stage2.yml"),
                               "threshold_profile_sha256": sha256(args.threshold_profile),
                               "defect_threshold": threshold}
            results.append(result)
            counts = Counter(slot["class"] for slot in result["slots"])
            summary_rows.append({"filename": path.name, "tray_verdict": result["tray_verdict"],
                                 "rule_tray_verdict": result["rule_tray_verdict"],
                                 "good_complete_order": result["groups"]["good"]["findings"]["complete_order"],
                                 "good_missing_vial": result["groups"]["good"]["findings"]["missing_vial"],
                                 "good_bad_as_good": result["groups"]["good"]["findings"]["bad_misclassified_as_good"],
                                 **{f"count_{name}": counts[name] for name in CLASSES}})
            for slot in result["slots"]:
                position_rows.append({"filename": path.name, "slot": slot["id"],
                                      "area": slot["area"], "position": slot["position"],
                                      "class": slot["class"], "presence": slot["presence"],
                                      "source": slot.get("prediction_source"),
                                      "defect_score": slot.get("defect_score"),
                                      "defect_threshold": slot.get("defect_threshold"),
                                      "review_status": slot["review_status"]})
            print(f"INFERRED {path.name} verdict={result['tray_verdict']} "
                  f"good_defects={len(result['groups']['good']['visible_defect_positions'])}")
    finally:
        empty_yolo.unlink(missing_ok=True)

    payload = {"schema_version": "vial-inference-1",
               "warning": "Predictions are model/appearance outputs, not human review.",
               "results": results}
    (args.output/"results.json").write_text(json.dumps(payload, indent=2)+"\n")
    for filename, rows in (("positions.csv", position_rows), ("summary.csv", summary_rows)):
        with (args.output/filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"INFERENCE VERIFIED images={len(results)} positions={len(position_rows)} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
