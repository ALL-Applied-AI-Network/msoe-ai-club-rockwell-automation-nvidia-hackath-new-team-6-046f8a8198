#!/usr/bin/env python3
"""Run leakage-free 60-position vial inspection and emit auditable tray verdicts."""
import argparse
from collections import Counter
import csv
import hashlib
from html import escape
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from urllib.parse import quote

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_real_dataset import CLASSES, extract_feature, load_predictor, predict
from package_model import BUNDLE_SCHEMA, COMPONENTS
from train_tray_classifier import VERDICTS, tray_feature_vector


CLASS_COLORS = {
    "pass": (70, 190, 70),
    "missing_cap": (45, 45, 230),
    "crooked_cap": (0, 165, 255),
    "damaged_vial": (190, 70, 190),
    "no_vial": (220, 140, 30),
}
CLASS_SHORT = {
    "pass": "P",
    "missing_cap": "MC",
    "crooked_cap": "CC",
    "damaged_vial": "DV",
    "no_vial": "NV",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_bundle(bundle):
    """Verify a deployment bundle and return its absolute component paths."""
    bundle = Path(bundle).resolve()
    manifest_path = bundle/"manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid bundle manifest: {exc}") from exc
    if manifest.get("schema_version") != BUNDLE_SCHEMA:
        raise ValueError("unsupported bundle schema")
    declared = manifest.get("components")
    if not isinstance(declared, dict):
        raise ValueError("bundle manifest has no component map")

    resolved = {}
    for name, expected_relative in COMPONENTS.items():
        entry = declared.get(name)
        if not isinstance(entry, dict) or entry.get("path") != expected_relative:
            raise ValueError(f"bundle component {name!r} is missing or misrouted")
    for name, entry in declared.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"bundle component {name!r} is invalid")
        relative = Path(entry["path"])
        if relative.is_absolute():
            raise ValueError(f"bundle component {name!r} has an absolute path")
        path = (bundle/relative).resolve()
        try:
            path.relative_to(bundle)
        except ValueError as exc:
            raise ValueError(f"bundle component {name!r} escapes bundle root") from exc
        if not path.is_file():
            raise ValueError(f"bundle component {name!r} is missing")
        if path.stat().st_size != entry.get("bytes"):
            raise ValueError(f"bundle component {name!r} has the wrong size")
        if sha256(path) != entry.get("sha256"):
            raise ValueError(f"bundle component {name!r} failed SHA-256 verification")
        resolved[name] = path

    calibration = json.loads(resolved["calibration"].read_text(encoding="utf-8"))
    declared_paths = {entry["path"]: name for name, entry in declared.items()}
    if len(declared_paths) != len(declared):
        raise ValueError("bundle manifest declares duplicate component paths")
    for profile in calibration.get("profiles", []):
        reference_name = profile.get("reference_image")
        if (not isinstance(reference_name, str) or
                Path(reference_name).name != reference_name):
            raise ValueError("bundle calibration has an invalid reference image")
        relative = f"config/examples/{reference_name}"
        component = declared_paths.get(relative)
        if component is None:
            raise ValueError(f"bundle calibration reference {reference_name!r} is missing")
        if sha256(resolved[component]) != profile.get("reference_sha256"):
            raise ValueError(f"bundle calibration reference {reference_name!r} has the wrong hash")
    resolved["manifest"] = manifest_path
    resolved["manifest_sha256"] = sha256(manifest_path)
    return resolved


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


def confidence_gated_verdict(model_verdict, complete_margin, threshold):
    """Only accept complete orders that clear the calibrated safety margin."""
    if model_verdict != "complete_order":
        return model_verdict
    return "complete_order" if complete_margin >= threshold else "unresolved"


def render_annotated(image, result, destination):
    """Render all mapped positions and both tray verdicts on a source image."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    for slot in result["slots"]:
        label = slot.get("class", "no_vial")
        color = CLASS_COLORS.get(label, (180, 180, 180))
        left, top, right, bottom = (int(value) for value in slot["bbox"])
        thickness = 2 if label == "pass" else 3
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)
        text = f"{slot['id']}:{CLASS_SHORT.get(label, '?')}"
        origin = (left, max(14, top-5))
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, color, 1, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(height, 108)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    model_verdict = result.get("tray_verdict", "unresolved").replace("_", " ").upper()
    rule_verdict = result.get("rule_tray_verdict", "unresolved").replace("_", " ")
    profile = result.get("alignment", {}).get("profile", "unknown")
    cv2.putText(canvas, f"{result['filename']}  |  MODEL: {model_verdict}",
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    gate = result.get("tray_confidence_gate")
    gate_text = ""
    if gate is not None:
        gate_text = (f"  |  Gate: {gate['complete_margin']:.3f} / "
                     f"{gate['complete_margin_threshold']:.3f}")
    cv2.putText(canvas, f"Rules: {rule_verdict}  |  Calibration: {profile}{gate_text}",
                (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (225, 225, 225), 1, cv2.LINE_AA)
    x = 20
    for label in CLASSES:
        color = CLASS_COLORS[label]
        cv2.rectangle(canvas, (x, 80), (x+16, 96), color, -1)
        text = label.replace("_", " ")
        cv2.putText(canvas, text, (x+22, 94), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 235, 235), 1, cv2.LINE_AA)
        x += 36 + int(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     0.42, 1)[0][0])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas,
                       [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write annotated image: {destination}")


def write_html_report(results, destination):
    """Write a portable, dependency-free visual summary of inference results."""
    verdict_counts = Counter(result["tray_verdict"] for result in results)
    cards = []
    for result in results:
        verdict = result["tray_verdict"]
        css_verdict = verdict if verdict in VERDICTS else "unresolved"
        counts = Counter(slot["class"] for slot in result["slots"])
        chips = "".join(
            f'<span class="chip {escape(name)}">{escape(name.replace("_", " "))}: '
            f'{counts[name]}</span>' for name in CLASSES
        )
        visible = result["groups"]["good"].get("visible_defect_positions", [])
        defect_text = ", ".join(escape(str(value)) for value in visible) or "none"
        image_path = quote(result["annotated_image"].replace("\\", "/"))
        gate = result.get("tray_confidence_gate")
        gate_html = ""
        if gate is not None:
            gate_html = (
                f' &nbsp; <strong>Base model:</strong> '
                f'{escape(result["tray_model_verdict"].replace("_", " "))}'
                f' &nbsp; <strong>Complete margin:</strong> '
                f'{gate["complete_margin"]:.3f} / '
                f'{gate["complete_margin_threshold"]:.3f}'
            )
        cards.append(f"""
        <article class="tray-card">
          <div class="card-heading">
            <h2>{escape(result['filename'])}</h2>
            <span class="verdict {css_verdict}">{escape(verdict.replace('_', ' '))}</span>
          </div>
          <p><strong>Inspection rules:</strong> {escape(result['rule_tray_verdict'].replace('_', ' '))}
             &nbsp; <strong>Calibration:</strong> {escape(str(result['alignment']['profile']))}
             {gate_html}</p>
          <a href="{image_path}"><img src="{image_path}" alt="Annotated {escape(result['filename'])}"></a>
          <div class="chips">{chips}</div>
          <p><strong>Good-tray visible defect positions:</strong> {defect_text}</p>
        </article>""")

    summary = "".join(
        f'<span class="summary-item"><strong>{verdict_counts[name]}</strong> '
        f'{escape(name.replace("_", " "))}</span>'
        for name in ("complete_order", "incomplete_order", "unresolved")
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vial inspection report</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; }}
    body {{ margin: 0; background: #0b1020; color: #e8edf7; }}
    header, main {{ width: min(1180px, calc(100% - 32px)); margin: auto; }}
    header {{ padding: 34px 0 20px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(1.8rem, 4vw, 3rem); }}
    .notice {{ color: #b9c3d8; max-width: 800px; }}
    .summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 20px; }}
    .summary-item, .chip {{ background: #172038; border: 1px solid #2b3858; border-radius: 999px; padding: 8px 12px; }}
    main {{ display: grid; gap: 22px; padding-bottom: 48px; }}
    .tray-card {{ background: #11182a; border: 1px solid #283553; border-radius: 16px; padding: 18px; box-shadow: 0 14px 34px #0005; }}
    .card-heading {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    h2 {{ margin: 0; font-size: 1.25rem; }}
    .verdict {{ border-radius: 8px; padding: 7px 10px; font-weight: 700; text-transform: uppercase; }}
    .complete_order {{ background: #174d31; color: #a9f5ca; }}
    .incomplete_order {{ background: #672628; color: #ffd0d0; }}
    .unresolved {{ background: #5a491d; color: #ffe4a0; }}
    img {{ display: block; width: 100%; height: auto; border-radius: 10px; border: 1px solid #34415d; background: #070a11; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .chip {{ padding: 5px 9px; font-size: .88rem; }}
    .pass {{ border-color: #46be46; }} .missing_cap {{ border-color: #e62d2d; }}
    .crooked_cap {{ border-color: #ffa500; }} .damaged_vial {{ border-color: #be46be; }}
    .no_vial {{ border-color: #1e8cdc; }} a {{ color: inherit; }}
  </style>
</head>
<body>
  <header>
    <h1>Vial inspection report</h1>
    <p class="notice">Automated predictions for demonstration and evaluation. Results are not human-reviewed production dispositions.</p>
    <div class="summary">{summary}</div>
  </header>
  <main>{''.join(cards)}</main>
</body>
</html>
"""
    Path(destination).write_text(document, encoding="utf-8")


class InspectionEngine:
    """Reusable inference engine that loads all model artifacts exactly once."""

    def __init__(self, labeler_path, calibration, model, threshold_profile,
                 tray_model=None, bundle=None):
        self.calibration = Path(calibration)
        self.model_path = Path(model)
        self.threshold_profile = Path(threshold_profile)
        self.tray_model_path = Path(tray_model) if tray_model is not None else None
        self.bundle = bundle

        profile = json.loads(self.threshold_profile.read_text())
        if profile.get("schema_version") not in {
                "vial-two-stage-threshold-1", "vial-tray-threshold-1"}:
            raise ValueError("unsupported threshold profile")
        self.threshold = float(profile["defect_threshold"])
        self.predictor = load_predictor(self.model_path, self.threshold)
        if self.predictor["kind"] != "two_stage":
            raise ValueError("model must be a two-stage model directory")

        self.tray_model = self.tray_scaler = self.tray_metadata = self.tray_gate = None
        self.tray_model_provenance = None
        if self.tray_model_path is not None:
            self.tray_metadata = json.loads(
                (self.tray_model_path/"metrics.json").read_text())
            if self.tray_metadata.get("schema_version") != "vial-tray-classifier-2":
                raise ValueError("unsupported tray classifier")
            self.tray_gate = self.tray_metadata.get("confidence_gate")
            if (not isinstance(self.tray_gate, dict) or
                    self.tray_gate.get("schema_version") !=
                    "vial-tray-confidence-gate-1"):
                raise ValueError("tray classifier has no supported confidence gate")
            self.tray_model = cv2.ml.SVM_load(
                str(self.tray_model_path/"model.yml"))
            with np.load(self.tray_model_path/"scaler.npz") as values:
                self.tray_scaler = (values["mean"], values["std"])
            self.tray_model_provenance = {
                "feature_set": self.tray_metadata["feature_set"],
                "model_sha256": sha256(self.tray_model_path/"model.yml"),
                "scaler_sha256": sha256(self.tray_model_path/"scaler.npz"),
            }

        self.labeler = load_module("vial_inference_labeler", labeler_path)
        self.prediction_module = __import__("prediction")
        self.rules = __import__("rules")
        self.hog = cv2.HOGDescriptor(
            (64, 64), (16, 16), (8, 8), (8, 8), 9)
        temporary = tempfile.NamedTemporaryFile(
            prefix="vial-empty-detections-", suffix=".txt", delete=False)
        temporary.close()
        self.empty_yolo = Path(temporary.name)
        self.model_provenance = {
            "directory": str(self.model_path.resolve()),
            "stage1_sha256": sha256(self.model_path/"stage1.yml"),
            "stage2_sha256": sha256(self.model_path/"stage2.yml"),
            "threshold_profile_sha256": sha256(self.threshold_profile),
            "defect_threshold": self.threshold,
        }

    def close(self):
        self.empty_yolo.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()

    def inspect(self, path, good_expected_through=30,
                scrap_expected_through=None):
        path = Path(path)
        prepared, _ = self.labeler.prepare([path], self.calibration)
        result = prepared[0]
        with Image.open(path) as source:
            pil_image = source.convert("RGB")
            # Empty detections intentionally activate appearance/fill fallback only.
            self.prediction_module.predict(
                result, pil_image, self.empty_yolo, auto_confirm=False)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode image: {path}")
        good = [slot for slot in result["slots"] if slot["id"].startswith("G")]
        features = np.asarray(
            [extract_feature(image, slot["bbox"], self.hog) for slot in good],
            dtype=np.float32)
        predicted, scores, defect_ids = predict(self.predictor, features)
        for index, (slot, class_id) in enumerate(zip(good, predicted)):
            label = CLASSES[int(class_id)]
            slot.update({
                "class": label, "presence": label != "no_vial",
                "suggested_class": label, "review_status": "predicted",
                "prediction_source": "two_stage_hog_svm",
                "prediction_confidence": None,
                "defect_score": round(float(scores[index]), 8),
                "defect_threshold": self.threshold,
                "defect_candidate": CLASSES[int(defect_ids[index])],
                "reason": "model:two_stage_hog_svm; score is not a probability",
            })
        for slot in result["slots"]:
            if slot["id"].startswith("S"):
                slot["review_status"] = "predicted"
        result["groups"] = {
            "good": self.rules.summarize(
                "good", good, good_expected_through),
            "scrap": self.rules.summarize(
                "scrap",
                [slot for slot in result["slots"] if slot["id"].startswith("S")],
                scrap_expected_through),
        }
        result["rule_tray_verdict"] = tray_verdict(result["groups"])
        if self.tray_model is not None:
            tray_features = tray_feature_vector(
                scores, defect_ids, result["alignment"]["profile"],
                self.tray_metadata["feature_set"])
            mean, std = self.tray_scaler
            scaled_tray = ((tray_features-mean)/std).reshape(
                1, -1).astype(np.float32)
            _, tray_prediction = self.tray_model.predict(scaled_tray)
            _, raw_margin = self.tray_model.predict(
                scaled_tray, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            model_verdict = VERDICTS[int(tray_prediction[0, 0])]
            complete_margin = (float(raw_margin[0, 0]) *
                               float(self.tray_gate["complete_margin_sign"]))
            complete_threshold = float(
                self.tray_gate["complete_margin_threshold"])
            result["tray_model_verdict"] = model_verdict
            result["tray_verdict"] = confidence_gated_verdict(
                model_verdict, complete_margin, complete_threshold)
            result["tray_confidence_gate"] = {
                "schema_version": self.tray_gate["schema_version"],
                "complete_margin": round(complete_margin, 8),
                "complete_margin_threshold": complete_threshold,
                "accepted_complete": result["tray_verdict"] == "complete_order",
                "low_confidence_action": "unresolved",
                "selected_on": self.tray_gate["selected_on"],
            }
            result["tray_model"] = dict(self.tray_model_provenance)
        else:
            result["tray_verdict"] = result["rule_tray_verdict"]
        result["backend"] = "two_stage_hog_svm_plus_scrap_appearance"
        result["model"] = dict(self.model_provenance)
        if self.bundle is not None:
            result["bundle"] = {
                "schema_version": BUNDLE_SCHEMA,
                "manifest_sha256": self.bundle["manifest_sha256"],
            }
        return result, image


def result_rows(result):
    """Create the stable tray and position CSV rows for one result."""
    counts = Counter(slot["class"] for slot in result["slots"])
    summary = {
        "filename": result["filename"],
        "tray_verdict": result["tray_verdict"],
        "tray_model_verdict": result.get("tray_model_verdict"),
        "rule_tray_verdict": result["rule_tray_verdict"],
        "complete_margin": result.get("tray_confidence_gate", {}).get(
            "complete_margin"),
        "complete_margin_threshold": result.get(
            "tray_confidence_gate", {}).get("complete_margin_threshold"),
        "route_action": result.get("service", {}).get("route_action"),
        "good_complete_order": result["groups"]["good"]["findings"][
            "complete_order"],
        "good_missing_vial": result["groups"]["good"]["findings"][
            "missing_vial"],
        "good_bad_as_good": result["groups"]["good"]["findings"][
            "bad_misclassified_as_good"],
        **{f"count_{name}": counts[name] for name in CLASSES},
    }
    positions = [{
        "filename": result["filename"], "slot": slot["id"],
        "area": slot["area"], "position": slot["position"],
        "class": slot["class"], "presence": slot["presence"],
        "source": slot.get("prediction_source"),
        "defect_score": slot.get("defect_score"),
        "defect_threshold": slot.get("defect_threshold"),
        "review_status": slot["review_status"],
    } for slot in result["slots"]]
    return summary, positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeler", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--files", required=True, nargs="+")
    deployment = parser.add_mutually_exclusive_group(required=True)
    deployment.add_argument("--bundle", type=Path,
                            help="verified bundle created by package_model.py")
    deployment.add_argument("--model", type=Path,
                            help="two-stage model directory")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--threshold-profile", type=Path)
    parser.add_argument("--tray-model", type=Path,
                        help="optional trained tray classifier directory")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-visual-report", action="store_true",
                        help="skip annotated images and report.html")
    parser.add_argument("--good-expected-through", type=int, default=30)
    parser.add_argument("--scrap-expected-through", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new directory")
    if any(Path(name).name != name for name in args.files):
        parser.error("files must be image basenames")
    bundle = None
    if args.bundle is not None:
        if any(value is not None for value in
               (args.calibration, args.threshold_profile, args.tray_model)):
            parser.error("bundle cannot be combined with calibration, threshold-profile, or tray-model")
        try:
            bundle = resolve_bundle(args.bundle)
        except ValueError as exc:
            parser.error(str(exc))
        args.model = bundle["position_metadata"].parent
        args.threshold_profile = bundle["threshold_profile"]
        args.tray_model = bundle["tray_metadata"].parent
        args.calibration = bundle["calibration"]
    elif args.calibration is None or args.threshold_profile is None:
        parser.error("direct model mode requires calibration and threshold-profile")

    paths = [args.images/name for name in args.files]
    for path in paths:
        if not path.is_file():
            parser.error(f"image does not exist: {path}")
    args.output.mkdir(parents=True)
    results, position_rows, summary_rows = [], [], []

    try:
        engine = InspectionEngine(
            args.labeler, args.calibration, args.model,
            args.threshold_profile, args.tray_model, bundle)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    with engine:
        for path in paths:
            result, image = engine.inspect(
                path, args.good_expected_through, args.scrap_expected_through)
            if not args.no_visual_report:
                relative_annotation = Path("annotated")/path.name
                render_annotated(image, result, args.output/relative_annotation)
                result["annotated_image"] = relative_annotation.as_posix()
            results.append(result)
            summary, positions = result_rows(result)
            summary_rows.append(summary)
            position_rows.extend(positions)
            print(f"INFERRED {path.name} verdict={result['tray_verdict']} "
                  f"good_defects={len(result['groups']['good']['visible_defect_positions'])}")

    payload = {"schema_version": "vial-inference-1",
               "warning": "Predictions are model/appearance outputs, not human review.",
               "results": results}
    (args.output/"results.json").write_text(json.dumps(payload, indent=2)+"\n")
    for filename, rows in (("positions.csv", position_rows), ("summary.csv", summary_rows)):
        with (args.output/filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if not args.no_visual_report:
        write_html_report(results, args.output/"report.html")
    print(f"INFERENCE VERIFIED images={len(results)} positions={len(position_rows)} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
