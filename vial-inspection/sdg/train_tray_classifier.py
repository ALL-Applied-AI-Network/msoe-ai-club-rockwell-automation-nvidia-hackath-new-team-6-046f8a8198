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

    combined_x = np.concatenate([train_x, val_x])
    combined_y = np.concatenate([train_y, val_y])
    test_x, test_y, test_names = partitions["Testing"]
    model, mean, std, test_prediction = train_predict(selected, combined_x, combined_y, test_x)
    test_metrics = metrics(test_y, test_prediction)
    args.output.mkdir(parents=True)
    model.save(str(args.output/"model.yml"))
    np.savez(args.output/"scaler.npz", mean=mean, std=std)
    report = {"schema_version": "vial-tray-classifier-1",
              "feature_set": args.feature_set,
              "feature_count": int(combined_x.shape[1]), "selected_config": selected,
              "split_counts": {split: {name: int(np.sum(partitions[split][1] == index))
                                        for index, name in enumerate(VERDICTS)}
                               for split in partitions},
              "validation_selection_metrics": validation_metrics,
              "testing_metrics": test_metrics, "validation_trials": trials,
              "testing_predictions": [{"image": name, "actual": VERDICTS[int(actual)],
                                       "predicted": VERDICTS[int(predicted)]}
                                      for name, actual, predicted in zip(
                                          test_names, test_y, test_prediction)]}
    (args.output/"metrics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(f"TRAY CLASSIFIER VERIFIED val_balanced={validation_metrics['balanced_accuracy']:.4f} "
          f"test_accuracy={test_metrics['accuracy']:.4f} "
          f"test_balanced={test_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
