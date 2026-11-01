import csv
import hashlib
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"export_dataset.py"
SPEC = importlib.util.spec_from_file_location("export_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_dataset(root):
    scene = root/"scene_00000"
    crops = scene/"crops"
    crops.mkdir(parents=True)
    slots = []
    for prefix in ("G", "S"):
        for number in range(1, 31):
            slot_id = f"{prefix}{number:02}"
            crop = crops/f"{slot_id}.png"
            crop.write_bytes(slot_id.encode())
            slots.append({"id": slot_id, "class": MODULE.CLASSES[(number-1) % 5],
                          "crop": f"crops/{slot_id}.png"})
    labels_path = scene/"labels.json"
    labels_path.write_text(json.dumps({"seed": 9, "slots": slots}))
    dataset = {"schema_version": "sdg-dataset-1", "scene_count": 1,
               "scenes": [{"id": "scene_00000", "split": "train",
                            "labels": "scene_00000/labels.json",
                            "labels_sha256": digest(labels_path)}]}
    (root/"dataset.json").write_text(json.dumps(dataset))


def test_export_good_positions_is_balanced(tmp_path):
    source, output = tmp_path/"source", tmp_path/"output"
    build_dataset(source)
    MODULE.export(source, output)
    metadata = json.loads((output/"dataset.json").read_text())
    assert metadata["sample_count"] == 30
    assert set(metadata["class_counts"]["train"].values()) == {6}
    with (output/"manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert all(row["slot"].startswith("G") for row in rows)


def test_export_all_positions_includes_scrap(tmp_path):
    source, output = tmp_path/"source", tmp_path/"output"
    build_dataset(source)
    MODULE.export(source, output, areas="all")
    assert json.loads((output/"dataset.json").read_text())["sample_count"] == 60
