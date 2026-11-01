import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1]/"train_tray_classifier.py"
SPEC = importlib.util.spec_from_file_location("train_tray_classifier", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tray_feature_vector_is_fixed_and_profile_aware():
    scores = np.arange(30, dtype=np.float32)/10
    first = MODULE.tray_feature_vector(scores, [1]*30, "view_001", "full")
    second = MODULE.tray_feature_vector(scores, [1]*30, "view_122", "full")
    assert first.shape == (81,)
    np.testing.assert_array_equal(first[:-4], second[:-4])
    np.testing.assert_array_equal(first[-4:], [1, 0, 0, 0])
    np.testing.assert_array_equal(second[-4:], [0, 1, 0, 0])
    distribution = MODULE.tray_feature_vector(scores, [1]*30, "view_001")
    assert distribution.shape == (47,)


def test_metrics_balances_rare_complete_class():
    result = MODULE.metrics([0, 1, 1, 1], [0, 0, 1, 1])
    assert result["accuracy"] == 0.75
    assert np.isclose(result["balanced_accuracy"], (1.0+2/3)/2)
