#!/usr/bin/env python3
"""Train a tray-level complete/incomplete classifier from position-model scores."""
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


VERDICTS = ("complete_order", "incomplete_order")
PROFILES = ("view_001", "view_122", "view_250_dark_upper", "view_271_bright_lower")


def load_labeler(path):
    path = Path(path).resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("vial_tray_training_labeler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tray_feature_vector(scores, defect_ids, profile, feature_set="distribution"):
    scores = np.asarray(scores, dtype=np.float32)
    if scores.shape != (30,):
        raise ValueError("expected 30 ordered good-tray scores")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 11)).astype(np.float32)
    defect_counts = np.bincount(np.asarray(defect_ids, dtype=int)-1, minlength=4).astype(np.float32)/30.0
    profile_one_hot = np.asarray([float(profile == name) for name in PROFILES], dtype=np.float32)
    distribution = [np.sort(scores), quantiles,
                    [scores.mean(), scores.std()], defect_counts]
    if feature_set == "distribution":
        return np.concatenate(distribution)
    if feature_set == "full":
        return np.concatenate([scores, *distribution, profile_one_hot])
    raise ValueError(f"unknown feature set: {feature_set}")


def metrics(actual, predicted):
    matrix = np.zeros((2, 2), dtype=int)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    recall = [matrix[index, index]/matrix[index].sum() if matrix[index].sum() else None
              for index in range(2)]
    supported = [value for value in recall if value is not None]
    return {"accuracy": float(np.trace(matrix)/matrix.sum()),
            "balanced_accuracy": float(sum(supported)/len(supported)),
            "recall": {name: recall[index] for index, name in enumerate(VERDICTS)},
            "confusion_matrix": matrix.tolist()}


def fit_scaler(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def make_svm(config, labels):
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR if config["kernel"] == "linear" else cv2.ml.SVM_RBF)
    svm.setC(config["c"])
    if config["kernel"] == "rbf":
        svm.setGamma(config["gamma"])
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = len(labels)/(2.0*counts)
    svm.setClassWeights(weights)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 5000, 1e-6))
    return svm


def train_predict(config, train_x, train_y, eval_x):
    mean, std = fit_scaler(train_x)
    svm = make_svm(config, train_y)
    svm.train((train_x-mean)/std, cv2.ml.ROW_SAMPLE, train_y)
    _, prediction = svm.predict((eval_x-mean)/std)
    return svm, mean, std, prediction.reshape(-1).astype(int)


def decision_margins(model, values, mean, std):
    """Return OpenCV's signed distance from the binary SVM boundary."""
    _, raw = model.predict((values-mean)/std,
                           flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
    return raw.reshape(-1).astype(np.float64)


def complete_margin_sign(model, train_x, train_y, mean, std):
    """Orient raw margins so larger values always mean more complete-like."""
    raw = decision_margins(model, train_x, mean, std)
    complete = raw[train_y == VERDICTS.index("complete_order")]
    incomplete = raw[train_y == VERDICTS.index("incomplete_order")]
    if not len(complete) or not len(incomplete):
        raise ValueError("both verdict classes are required to orient margins")
    return 1.0 if float(complete.mean()) > float(incomplete.mean()) else -1.0


def select_complete_margin_threshold(actual, complete_margins):
    """Select the lowest nonnegative threshold with zero validation false accepts."""
    actual = np.asarray(actual, dtype=int)
    complete_margins = np.asarray(complete_margins, dtype=np.float64)
    incomplete = complete_margins[actual == VERDICTS.index("incomplete_order")]
    if not len(incomplete):
        raise ValueError("validation requires incomplete-order examples")
    return float(max(0.0, np.nextafter(incomplete.max(), np.inf)))


def gated_predictions(base_predictions, complete_margins, threshold):
    """Treat low-confidence complete predictions as non-accepts for metrics."""
    base_predictions = np.asarray(base_predictions, dtype=int)
    accepted = ((base_predictions == VERDICTS.index("complete_order")) &
                (np.asarray(complete_margins) >= threshold))
    return np.where(accepted, VERDICTS.index("complete_order"),
                    VERDICTS.index("incomplete_order")).astype(int)


def gate_audit(names, actual, base_predictions, complete_margins, threshold):
    gated = gated_predictions(base_predictions, complete_margins, threshold)
    false_accepts = [name for name, truth, guess in zip(names, actual, gated)
                     if truth == 1 and guess == 0]
    false_rejects = [name for name, truth, guess in zip(names, actual, gated)
                     if truth == 0 and guess == 1]
    return {
        "metrics_treating_unresolved_as_incomplete": metrics(actual, gated),
        "accepted_complete": int(np.sum(gated == 0)),
        "false_accepts": false_accepts,
        "false_rejects_or_unresolved": false_rejects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-model", required=True, type=Path)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--verdicts", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--feature-set", choices=("distribution", "full"),
                        default="distribution")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    verdicts = {row["filename"]: row["verdict"]
                for row in csv.DictReader(args.verdicts.open(newline="", encoding="utf-8"))}
    labeler = load_labeler(args.labeler)
    predictor = load_predictor(args.position_model, 0.0)
    if predictor["kind"] != "two_stage":
        parser.error("position-model must be a two-stage directory")
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    partitions = {}
    failures = []
    for split in ("Training", "Validation", "Testing"):
        features, labels, names = [], [], []
        paths = sorted((args.dataset_root/split).glob("*.jpg"))
        for index, path in enumerate(paths, 1):
            try:
                verdict = verdicts[path.name]
                if verdict not in VERDICTS:
                    raise ValueError(f"unsupported verdict: {verdict}")
                result = labeler.prepare([path], args.calibration)[0][0]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                good = [slot for slot in result["slots"] if slot["id"].startswith("G")]
                crop_features = np.asarray([extract_feature(image, slot["bbox"], hog)
                                            for slot in good], dtype=np.float32)
                _, scores, defect_ids = predict(predictor, crop_features)
                features.append(tray_feature_vector(scores, defect_ids,
                                                    result["alignment"]["profile"],
                                                    args.feature_set))
                labels.append(VERDICTS.index(verdict))
                names.append(path.name)
                print(f"{split} [{index}/{len(paths)}] {path.name} {verdict}", flush=True)
            except (ValueError, KeyError, TypeError, OSError) as error:
                failures.append({"split": split, "image": path.name, "error": str(error)})
        partitions[split] = (np.asarray(features, dtype=np.float32),
                             np.asarray(labels, dtype=np.int32), names)
    if failures:
        raise SystemExit(f"feature extraction failures: {failures}")
    if any(len(np.unique(partitions[split][1])) != 2 for split in partitions):
        raise SystemExit("every split must contain both tray verdicts")

    candidates = ([{"kernel": "linear", "c": value} for value in (0.01, 0.1, 1.0, 10.0)]
                  + [{"kernel": "rbf", "c": c, "gamma": gamma}
                     for c in (0.1, 1.0, 10.0) for gamma in (0.001, 0.01, 0.1)])
    train_x, train_y, _ = partitions["Training"]
    val_x, val_y, _ = partitions["Validation"]
    trials, best = [], None
    for config in candidates:
        _, _, _, predicted = train_predict(config, train_x, train_y, val_x)
        result = metrics(val_y, predicted)
        trial = {"config": config, "metrics": result}
        trials.append(trial)
        key = (result["balanced_accuracy"], result["accuracy"],
               -config["c"], -config.get("gamma", 0.0))
        if best is None or key > best[0]:
            best = (key, config, result)
    _, selected, validation_metrics = best

    # Keep Validation outside the saved model: its raw margins calibrate the
    # fail-safe complete-order acceptance threshold used at inference time.
    model, mean, std, val_prediction = train_predict(
        selected, train_x, train_y, val_x)
    sign = complete_margin_sign(model, train_x, train_y, mean, std)
    val_margins = decision_margins(model, val_x, mean, std)*sign
    complete_threshold = select_complete_margin_threshold(val_y, val_margins)
    _, _, val_names = partitions["Validation"]
    validation_gate = gate_audit(
        val_names, val_y, val_prediction, val_margins, complete_threshold)

    test_x, test_y, test_names = partitions["Testing"]
    _, test_values = model.predict((test_x-mean)/std)
    test_prediction = test_values.reshape(-1).astype(int)
    test_margins = decision_margins(model, test_x, mean, std)*sign
    test_metrics = metrics(test_y, test_prediction)
    testing_gate = gate_audit(
        test_names, test_y, test_prediction, test_margins, complete_threshold)
    args.output.mkdir(parents=True)
    model.save(str(args.output/"model.yml"))
    np.savez(args.output/"scaler.npz", mean=mean, std=std)
    report = {"schema_version": "vial-tray-classifier-2",
              "feature_set": args.feature_set,
              "feature_count": int(train_x.shape[1]), "selected_config": selected,
              "training_policy": (
                  "Model fit on Training only; Validation selects configuration "
                  "and confidence gate; Testing is evaluation only."
              ),
              "split_counts": {split: {name: int(np.sum(partitions[split][1] == index))
                                        for index, name in enumerate(VERDICTS)}
                               for split in partitions},
              "validation_selection_metrics": validation_metrics,
              "confidence_gate": {
                  "schema_version": "vial-tray-confidence-gate-1",
                  "complete_margin_sign": sign,
                  "complete_margin_threshold": complete_threshold,
                  "selected_on": "Validation",
                  "selection_objective": (
                      "zero incomplete-order false accepts, then lowest threshold"
                  ),
                  "validation": validation_gate,
                  "testing": testing_gate,
              },
              "testing_metrics_before_gate": test_metrics,
              "validation_trials": trials,
              "testing_predictions": [{"image": name, "actual": VERDICTS[int(actual)],
                                       "model_prediction": VERDICTS[int(predicted)],
                                       "complete_margin": float(margin),
                                       "accepted_complete": bool(
                                           predicted == 0 and
                                           margin >= complete_threshold)}
                                      for name, actual, predicted, margin in zip(
                                          test_names, test_y, test_prediction,
                                          test_margins)]}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"TRAY CLASSIFIER VERIFIED val_balanced={validation_metrics['balanced_accuracy']:.4f} "
          f"test_accuracy={test_metrics['accuracy']:.4f} "
          f"test_balanced={test_metrics['balanced_accuracy']:.4f} "
          f"gate={complete_threshold:.6f} "
          f"test_false_accepts={len(testing_gate['false_accepts'])}")


if __name__ == "__main__":
    main()
