import importlib.util
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).parents[1]/"infer_tray.py"
SPEC = importlib.util.spec_from_file_location("infer_tray", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tray_verdict_preserves_three_valued_rule():
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": True}}}) == "complete_order"
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": False}}}) == "incomplete_order"
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": None}}}) == "unresolved"


def test_confidence_gate_never_promotes_incomplete_and_rejects_weak_complete():
    assert MODULE.confidence_gated_verdict("incomplete_order", 9.0, 0.4) == "incomplete_order"
    assert MODULE.confidence_gated_verdict("complete_order", 0.39, 0.4) == "unresolved"
    assert MODULE.confidence_gated_verdict("complete_order", 0.40, 0.4) == "complete_order"


def sample_result(filename="tray.jpg"):
    return {
        "filename": filename,
        "tray_verdict": "incomplete_order",
        "rule_tray_verdict": "incomplete_order",
        "alignment": {"profile": "view_test"},
        "annotated_image": "annotated/tray.jpg",
        "slots": [{
            "id": "G01", "class": "missing_cap", "bbox": [30, 115, 75, 145]
        }],
        "groups": {"good": {"visible_defect_positions": [1]}},
    }


def test_render_annotated_writes_overlay(tmp_path):
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    output = tmp_path/"annotated.jpg"

    MODULE.render_annotated(image, sample_result(), output)

    rendered = cv2.imread(str(output))
    assert rendered.shape == image.shape
    assert np.any(rendered != image)


def test_html_report_escapes_content_and_links_annotation(tmp_path):
    output = tmp_path/"report.html"

    MODULE.write_html_report([sample_result("tray_<unsafe>.jpg")], output)

    content = output.read_text(encoding="utf-8")
    assert "tray_&lt;unsafe&gt;.jpg" in content
    assert 'src="annotated/tray.jpg"' in content
    assert "incomplete order" in content
