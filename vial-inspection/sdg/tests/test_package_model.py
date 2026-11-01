import importlib.util
import json
from pathlib import Path

import pytest


SDG = Path(__file__).parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PACKAGE = load_module("package_model_test", SDG/"package_model.py")
INFER = load_module("infer_tray_bundle_test", SDG/"infer_tray.py")


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def build_inputs(tmp_path):
    position = tmp_path/"position-source"
    tray = tmp_path/"tray-source"
    position.mkdir()
    tray.mkdir()
    write_json(position/"model.json", {"schema_version": "vial-two-stage-1"})
    (position/"stage1.yml").write_bytes(b"gate")
    (position/"stage2.yml").write_bytes(b"defects")
    write_json(tray/"metrics.json", {
        "schema_version": "vial-tray-classifier-2",
        "feature_set": "distribution",
        "feature_count": 47,
        "confidence_gate": {
            "schema_version": "vial-tray-confidence-gate-1",
            "complete_margin_sign": 1.0,
            "complete_margin_threshold": 0.4,
        },
    })
    (tray/"model.yml").write_bytes(b"tray")
    (tray/"scaler.npz").write_bytes(b"scale")
    threshold = tmp_path/"threshold.json"
    write_json(threshold, {
        "schema_version": "vial-two-stage-threshold-1",
        "defect_threshold": 0.4,
    })
    calibration = tmp_path/"calibration.json"
    examples = tmp_path/"examples"
    examples.mkdir()
    reference = examples/"reference.jpg"
    reference.write_bytes(b"reference-image")
    write_json(calibration, {
        "schema_version": "vial-calibration-1",
        "profiles": [{
            "id": "view_test",
            "reference_image": reference.name,
            "reference_sha256": PACKAGE.sha256(reference),
        }],
    })
    return position, tray, threshold, calibration


def test_package_and_resolve_verified_bundle(tmp_path):
    inputs = build_inputs(tmp_path)
    output = tmp_path/"bundle"
    manifest = PACKAGE.package_bundle(*inputs, output)

    assert manifest["schema_version"] == PACKAGE.BUNDLE_SCHEMA
    assert len(manifest["components"]) == 9
    resolved = INFER.resolve_bundle(output)
    assert resolved["position_stage1"].read_bytes() == b"gate"
    assert resolved["tray_scaler"].read_bytes() == b"scale"
    assert resolved["calibration_reference_view_test"].read_bytes() == b"reference-image"


def test_resolve_bundle_rejects_tampered_weight(tmp_path):
    inputs = build_inputs(tmp_path)
    output = tmp_path/"bundle"
    PACKAGE.package_bundle(*inputs, output)
    (output/"position"/"stage1.yml").write_bytes(b"fake")

    with pytest.raises(ValueError, match="SHA-256 verification"):
        INFER.resolve_bundle(output)


def test_package_rejects_existing_output(tmp_path):
    inputs = build_inputs(tmp_path)
    output = tmp_path/"bundle"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        PACKAGE.package_bundle(*inputs, output)


def test_package_rejects_stale_calibration_reference(tmp_path):
    inputs = build_inputs(tmp_path)
    (tmp_path/"examples"/"reference.jpg").write_bytes(b"changed-reference")

    with pytest.raises(ValueError, match="reference hash mismatch"):
        PACKAGE.package_bundle(*inputs, tmp_path/"bundle")
