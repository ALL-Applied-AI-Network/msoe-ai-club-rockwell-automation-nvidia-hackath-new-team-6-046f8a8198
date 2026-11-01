import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import export_training


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.review = self.root / "review"
        (self.review / "images").mkdir(parents=True)
        self.source = self.review / "images" / "tray_000001.jpg"
        Image.new("RGB", (1920, 1080), "white").save(self.source)
        for directory in ("Training", "Validation", "Testing"):
            (self.root / "splits" / directory).mkdir(parents=True)
        (self.root / "splits" / "Training" / self.source.name).write_bytes(self.source.read_bytes())
        classes = export_training.CLASSES
        slots = []
        for prefix, area in (("G", "good"), ("S", "scrap")):
            for position in range(1, 31):
                slots.append({
                    "id": f"{prefix}{position:02d}", "area": area, "position": position,
                    "class": classes[(position - 1) % len(classes)], "review_status": "confirmed",
                    "bbox": [100, 100, 180, 190],
                })
        self.payload = {"results": [{
            "filename": self.source.name, "source_path": "images/" + self.source.name,
            "source_sha256": hashlib.sha256(self.source.read_bytes()).hexdigest(),
            "image_size": [1920, 1080], "slots": slots,
        }]}
        self.review_json = self.review / "labels.json"
        self.review_json.write_text(json.dumps(self.payload), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_both_formats_are_complete(self):
        output = self.root / "out"
        export_training.export(self.review_json, self.root / "splits", output, "both")
        self.assertEqual(len(list((output / "crops").rglob("*.jpg"))), 60)
        yolo = (output / "yolo" / "labels" / "train" / "tray_000001.txt").read_text().splitlines()
        self.assertEqual(len(yolo), 60)
        self.assertEqual(len((output / "manifest.csv").read_text().splitlines()), 61)

    def test_partial_review_is_rejected_without_output(self):
        payload = copy.deepcopy(self.payload)
        payload["results"][0]["slots"][0]["review_status"] = "draft"
        self.review_json.write_text(json.dumps(payload), encoding="utf-8")
        output = self.root / "out"
        with self.assertRaisesRegex(ValueError, "all positions must be confirmed"):
            export_training.export(self.review_json, self.root / "splits", output, "both")
        self.assertFalse(output.exists())

    def test_changed_source_is_rejected(self):
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source image hash changed"):
            export_training.export(self.review_json, self.root / "splits", self.root / "out", "crops")


if __name__ == "__main__":
    unittest.main()

