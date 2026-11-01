import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"evaluate_real.py"
SPEC = importlib.util.spec_from_file_location("evaluate_real", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_agreement_summary_counts_empty_and_present_classes():
    rows = [
        {"reference_class": "pass", "predicted_class": "pass"},
        {"reference_class": "pass", "predicted_class": "missing_cap"},
        {"reference_class": "no_vial", "predicted_class": "no_vial"},
    ]
    agreement, by_reference = MODULE.agreement_summary(rows)
    assert agreement == 2/3
    assert by_reference["pass"] == {"count": 2, "agreement": 0.5}
    assert by_reference["no_vial"] == {"count": 1, "agreement": 1.0}
    assert by_reference["crooked_cap"] == {"count": 0, "agreement": None}
