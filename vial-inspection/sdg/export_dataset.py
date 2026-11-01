#!/usr/bin/env python3
"""Validate an SDG dataset and export crops to an ImageFolder training layout."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export(dataset_root, output, areas="good", copy_mode="copy"):
    dataset_root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output already exists; choose a new directory")
    dataset = load_json(dataset_root/"dataset.json")
    if dataset.get("schema_version") != "sdg-dataset-1":
        raise ValueError("unsupported or missing dataset schema")
    scenes = dataset.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != dataset.get("scene_count"):
        raise ValueError("scene inventory does not match scene_count")
    if areas not in {"good", "all"}:
        raise ValueError("areas must be 'good' or 'all'")
    if copy_mode not in {"copy", "hardlink"}:
        raise ValueError("copy_mode must be 'copy' or 'hardlink'")

    prepared = []
    for scene in scenes:
        if scene.get("split") not in {"train", "val"}:
            raise ValueError(f"{scene.get('id')}: invalid split")
        labels_path = dataset_root/scene["labels"]
        if not labels_path.is_file() or sha256(labels_path) != scene.get("labels_sha256"):
            raise ValueError(f"{scene.get('id')}: labels missing or hash mismatch")
        labels = load_json(labels_path)
        slots = labels.get("slots")
        if not isinstance(slots, list) or len(slots) != 60:
            raise ValueError(f"{scene.get('id')}: expected 60 slots")
        for slot in slots:
            slot_id, label = slot.get("id"), slot.get("class")
            if label not in CLASSES or not isinstance(slot_id, str):
                raise ValueError(f"{scene.get('id')}: invalid slot label")
            if areas == "good" and not slot_id.startswith("G"):
                continue
            crop = labels_path.parent/slot["crop"]
            if not crop.is_file():
                raise ValueError(f"{scene.get('id')}/{slot_id}: crop missing")
            prepared.append((scene, labels, slot, crop))

    output.mkdir(parents=True)
    rows = []
    counts = {split: {name: 0 for name in CLASSES} for split in ("train", "val")}
    for scene, labels, slot, source in prepared:
        split, label = scene["split"], slot["class"]
        filename = f"{scene['id']}__{slot['id']}.png"
        relative = Path(split)/label/filename
        destination = output/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_mode == "hardlink":
            destination.hardlink_to(source)
        else:
            shutil.copyfile(source, destination)
        counts[split][label] += 1
        rows.append({"split": split, "class": label, "class_id": CLASSES.index(label),
                     "scene": scene["id"], "scene_seed": labels["seed"],
                     "slot": slot["id"], "image": relative.as_posix(),
                     "source_sha256": sha256(source)})

    with (output/"manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"schema_version": "vial-imagefolder-1", "classes": list(CLASSES),
                "areas": areas, "sample_count": len(rows), "class_counts": counts,
                "source_dataset_sha256": sha256(dataset_root/"dataset.json")}
    (output/"dataset.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print(f"EXPORT VERIFIED samples={len(rows)} areas={areas} output={output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--areas", choices=("good", "all"), default="good")
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    args = parser.parse_args()
    try:
        export(args.dataset_root, args.output, args.areas, args.copy_mode)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
