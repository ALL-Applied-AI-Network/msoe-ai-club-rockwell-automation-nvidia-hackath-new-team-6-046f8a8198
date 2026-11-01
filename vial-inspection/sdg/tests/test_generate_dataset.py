from collections import Counter
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"generate_dataset.py"
SPEC = importlib.util.spec_from_file_location("generate_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_states_has_all_positions_and_classes():
    states = MODULE.balanced_states(2026)
    assert set(states) == {f"G{index:02}" for index in range(1, 31)}
    assert Counter(states.values()) == Counter({name: 6 for name in MODULE.CLASSES})


def test_balanced_states_is_reproducible_and_seeded():
    assert MODULE.balanced_states(17) == MODULE.balanced_states(17)
    assert MODULE.balanced_states(17) != MODULE.balanced_states(18)
