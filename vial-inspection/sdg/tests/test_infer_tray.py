import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"infer_tray.py"
SPEC = importlib.util.spec_from_file_location("infer_tray", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tray_verdict_preserves_three_valued_rule():
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": True}}}) == "complete_order"
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": False}}}) == "incomplete_order"
    assert MODULE.tray_verdict({"good": {"findings": {"complete_order": None}}}) == "unresolved"
