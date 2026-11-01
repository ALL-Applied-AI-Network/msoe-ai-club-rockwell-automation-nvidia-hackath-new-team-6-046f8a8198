import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"calibrate_tray_verdict.py"
SPEC = importlib.util.spec_from_file_location("calibrate_tray_verdict", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_threshold_selection_separates_complete_and_incomplete():
    records = [
        {"actual": "complete_order", "max_defect_score": 0.1},
        {"actual": "complete_order", "max_defect_score": 0.2},
        {"actual": "incomplete_order", "max_defect_score": 0.8},
        {"actual": "incomplete_order", "max_defect_score": 0.9},
    ]
    threshold, metrics, count = MODULE.select_threshold(records)
    assert 0.2 < threshold < 0.8
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert count == 5
