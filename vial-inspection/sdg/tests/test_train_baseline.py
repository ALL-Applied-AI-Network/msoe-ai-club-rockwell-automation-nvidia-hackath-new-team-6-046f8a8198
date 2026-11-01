import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1]/"train_baseline.py"
SPEC = importlib.util.spec_from_file_location("train_baseline", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confusion_matrix_and_metrics():
    matrix = MODULE.confusion_matrix([0, 0, 1, 1], [0, 1, 1, 1], 2)
    np.testing.assert_array_equal(matrix, [[1, 1], [0, 2]])
    accuracy, per_class, macro_f1 = MODULE.metrics_from_confusion(matrix)
    assert accuracy == 0.75
    assert per_class[0]["recall"] == 0.5
    assert per_class[1]["recall"] == 1.0
    assert 0.7 < macro_f1 < 0.75
