#!/usr/bin/env python3
"""Package the vial inspection models and calibration into a verified bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


BUNDLE_SCHEMA = "vial-inspection-bundle-1"
COMPONENTS = {
    "position_metadata": "position/model.json",
    "position_stage1": "position/stage1.yml",
    "position_stage2": "position/stage2.yml",
    "threshold_profile": "config/two_stage_profile.json",
    "tray_metadata": "tray/metrics.json",
    "tray_model": "tray/model.yml",
    "tray_scaler": "tray/scaler.npz",
    "calibration": "config/calibration.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_schema(path, expected):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if payload.get("schema_version") != expected:
        raise ValueError(
            f"unsupported schema in {path}: expected {expected}, "
            f"got {payload.get('schema_version')!r}"
        )
    return payload


def package_bundle(position_model, tray_model, threshold_profile, calibration,
                   output):
    position_model, tray_model = Path(position_model), Path(tray_model)
    threshold_profile, calibration = Path(threshold_profile), Path(calibration)
    output = Path(output)
    if output.exists():
        raise ValueError("output already exists; choose a new directory")

    position_metadata = read_schema(position_model/"model.json", "vial-two-stage-1")
    threshold_metadata = read_schema(
        threshold_profile, "vial-two-stage-threshold-1")
    tray_metadata = read_schema(tray_model/"metrics.json", "vial-tray-classifier-1")
    try:
        calibration_metadata = json.loads(calibration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid calibration JSON {calibration}: {exc}") from exc

    sources = {
        "position_metadata": position_model/"model.json",
        "position_stage1": position_model/"stage1.yml",
        "position_stage2": position_model/"stage2.yml",
        "threshold_profile": threshold_profile,
        "tray_metadata": tray_model/"metrics.json",
        "tray_model": tray_model/"model.yml",
        "tray_scaler": tray_model/"scaler.npz",
        "calibration": calibration,
    }
    destinations = dict(COMPONENTS)
    calibration_hashes = {}
    seen_references = set()
    for profile in calibration_metadata.get("profiles", []):
        profile_id = profile.get("id")
        reference_name = profile.get("reference_image")
        if (not isinstance(profile_id, str) or not profile_id or
                not isinstance(reference_name, str) or
                Path(reference_name).name != reference_name):
            raise ValueError("calibration has an invalid profile reference")
        if reference_name in seen_references:
            continue
        seen_references.add(reference_name)
        component = f"calibration_reference_{profile_id}"
        if component in sources:
            raise ValueError(f"duplicate calibration profile id: {profile_id}")
        sources[component] = calibration.parent/"examples"/reference_name
        destinations[component] = f"config/examples/{reference_name}"
        calibration_hashes[component] = profile.get("reference_sha256")
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise ValueError("missing bundle inputs: " + ", ".join(missing))
    for component, expected_hash in calibration_hashes.items():
        if not isinstance(expected_hash, str) or sha256(sources[component]) != expected_hash:
            raise ValueError(
                f"calibration reference hash mismatch: {sources[component]}")

    output.mkdir(parents=True)
    try:
        files = {}
        for name, relative in destinations.items():
            destination = output/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources[name], destination)
            files[name] = {
                "path": relative,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        manifest = {
            "schema_version": BUNDLE_SCHEMA,
            "components": files,
            "runtime": {
                "position_model_schema": position_metadata["schema_version"],
                "tray_model_schema": tray_metadata["schema_version"],
                "threshold_schema": threshold_metadata["schema_version"],
                "defect_threshold": float(threshold_metadata["defect_threshold"]),
                "tray_feature_set": tray_metadata["feature_set"],
                "tray_feature_count": int(tray_metadata["feature_count"]),
                "calibration_schema": calibration_metadata.get("schema_version"),
            },
            "warning": (
                "Model predictions are not human review. Verify this bundle on the "
                "target camera and reject it if manifest verification fails."
            ),
        }
        (output/"manifest.json").write_text(
            json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-model", required=True, type=Path)
    parser.add_argument("--tray-model", required=True, type=Path)
    parser.add_argument("--threshold-profile", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = package_bundle(
            args.position_model, args.tray_model, args.threshold_profile,
            args.calibration, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    total_bytes = sum(item["bytes"] for item in manifest["components"].values())
    print(f"BUNDLE VERIFIED files={len(manifest['components'])} "
          f"bytes={total_bytes} output={args.output}")


if __name__ == "__main__":
    main()
