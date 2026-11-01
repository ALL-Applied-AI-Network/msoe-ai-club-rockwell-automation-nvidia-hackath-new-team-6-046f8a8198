#!/usr/bin/env python3
"""Export fully reviewed vial positions to classification crops and/or YOLO."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
MAPPING = json.loads((HERE / "class_mapping.json").read_text(encoding="utf-8"))
CLASSES = MAPPING["canonical_classes"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
SPLIT_NAMES = {"Training": "train", "Validation": "val", "Testing": "test"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_path(review_path, result):
    base = review_path.resolve().parent
    relative = Path(result["source_path"])
    candidate = (base / relative).resolve()
    if relative.is_absolute() or not candidate.is_relative_to(base):
        raise ValueError(f"{result.get('filename')}: source path escapes the review folder")
    if not candidate.is_file():
        raise ValueError(f"{result.get('filename')}: source image is missing")
    if sha256(candidate) != result.get("source_sha256"):
        raise ValueError(f"{result.get('filename')}: source image hash changed")
    return candidate


def split_index(root):
    index = {}
    root = Path(root)
    for directory, split in SPLIT_NAMES.items():
        folder = root / directory
        if not folder.is_dir():
            raise ValueError(f"Missing split directory: {folder}")
        for image in folder.iterdir():
            if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if image.name in index:
                    raise ValueError(f"Duplicate image basename across splits: {image.name}")
                index[image.name] = split
    return index


def validate_result(result):
    filename = result.get("filename", "<unknown>")
    if result.get("image_size") != [1920, 1080]:
        raise ValueError(f"{filename}: expected image_size [1920, 1080]")
    slots = result.get("slots")
    if not isinstance(slots, list) or len(slots) != 60:
        raise ValueError(f"{filename}: exactly 60 reviewed positions are required")
    if len({slot.get("id") for slot in slots}) != 60:
        raise ValueError(f"{filename}: position IDs must be unique")
    expected = {f"{prefix}{number:02d}" for prefix in ("G", "S") for number in range(1, 31)}
    if {slot.get("id") for slot in slots} != expected:
        raise ValueError(f"{filename}: position IDs do not match G01-G30 and S01-S30")
    for slot in slots:
        label = slot.get("class")
        if slot.get("review_status") != "confirmed" or label not in CLASS_TO_ID:
            raise ValueError(f"{filename}/{slot.get('id')}: all positions must be confirmed with a canonical class")
        box = slot.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or not all(type(value) is int for value in box):
            raise ValueError(f"{filename}/{slot.get('id')}: invalid bounding box")
        left, top, right, bottom = box
        if not (0 <= left < right <= 1920 and 0 <= top < bottom <= 1080):
            raise ValueError(f"{filename}/{slot.get('id')}: bounding box is outside the image")


def normalized_yolo(slot):
    left, top, right, bottom = slot["bbox"]
    x = (left + right) / 2 / 1920
    y = (top + bottom) / 2 / 1080
    width = (right - left) / 1920
    height = (bottom - top) / 1080
    return f"{CLASS_TO_ID[slot['class']]} {x:.8f} {y:.8f} {width:.8f} {height:.8f}"


def export(review, split_root, output, export_format):
    review = Path(review)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    payload = load_json(review)
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("Review contains no images")
    if len({result.get("filename") for result in results}) != len(results):
        raise ValueError("Review contains duplicate image names")
    splits = split_index(split_root)
    prepared = []
    for result in results:
        validate_result(result)
        if result["filename"] not in splits:
            raise ValueError(f"{result['filename']}: image is absent from the supplied split root")
        prepared.append((result, source_path(review, result), splits[result["filename"]]))

    output.mkdir(parents=True)
    rows = []
    for result, source, split in prepared:
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            if image.size != (1920, 1080):
                raise ValueError(f"{result['filename']}: decoded image size changed")
            stem = Path(result["filename"]).stem
            if export_format in {"yolo", "both"}:
                image_dir = output / "yolo" / "images" / split
                label_dir = output / "yolo" / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, image_dir / result["filename"])
                lines = [normalized_yolo(slot) for slot in result["slots"]]
                (label_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            for slot in result["slots"]:
                crop_relative = Path(split) / slot["class"] / f"{stem}__{slot['id']}.jpg"
                if export_format in {"crops", "both"}:
                    crop_path = output / "crops" / crop_relative
                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(slot["bbox"]).save(crop_path, quality=95)
                rows.append({
                    "split": split,
                    "image": result["filename"],
                    "slot": slot["id"],
                    "area": slot["area"],
                    "position": slot["position"],
                    "class": slot["class"],
                    "class_id": CLASS_TO_ID[slot["class"]],
                    "bbox_xyxy": " ".join(map(str, slot["bbox"])),
                    "crop": str(crop_relative).replace("\\", "/") if export_format in {"crops", "both"} else "",
                    "source_sha256": result["source_sha256"],
                })

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "classes.json").write_text(json.dumps({"classes": CLASSES}, indent=2) + "\n", encoding="utf-8")
    if export_format in {"yolo", "both"}:
        yaml = ["path: .", "train: images/train", "val: images/val", "test: images/test", "names:"]
        yaml.extend(f"  {index}: {name}" for index, name in enumerate(CLASSES))
        (output / "yolo" / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"Exported {len(prepared)} images and {len(rows)} confirmed positions to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True, type=Path, help="Imported/confirmed labeler labels.json")
    parser.add_argument("--split-root", required=True, type=Path, help="Dataset images folder containing Training/Validation/Testing")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--format", choices=("crops", "yolo", "both"), default="both")
    args = parser.parse_args()
    try:
        export(args.review, args.split_root, args.output, args.format)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

