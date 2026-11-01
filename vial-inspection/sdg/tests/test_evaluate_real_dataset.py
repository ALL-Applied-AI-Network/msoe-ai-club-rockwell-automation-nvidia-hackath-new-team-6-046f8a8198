import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"evaluate_real_dataset.py"
SPEC = importlib.util.spec_from_file_location("evaluate_real_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summarize_reports_macro_and_unsupported_class():
    rows = [
        {"reference_class": "pass", "predicted_class": "pass"},
        {"reference_class": "pass", "predicted_class": "no_vial"},
        {"reference_class": "no_vial", "predicted_class": "no_vial"},
    ]
    result = MODULE.summarize(rows)
    assert result["agreement"] == 2/3
    assert result["macro_class_agreement"] == 0.75
    assert result["per_class"]["pass"]["agreement"] == 0.5
    assert result["per_class"]["missing_cap"]["agreement"] is None
