import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"build_real_pseudo_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_real_pseudo_dataset", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_balanced_select_uses_smallest_class_per_split():
    candidates = []
    for split in ("train", "val"):
        for class_index, name in enumerate(MODULE.CLASSES):
            for item_index in range(class_index+2):
                candidates.append({"split": split, "class": name,
                                   "image": f"image_{item_index}.jpg",
                                   "slot": f"G{item_index+1:02}"})
    selected = MODULE.balanced_select(candidates, seed=4, maximum=10)
    for split in ("train", "val"):
        values = [item for item in selected if item["split"] == split]
        assert len(values) == 10
        assert {name: sum(item["class"] == name for item in values)
                for name in MODULE.CLASSES} == {name: 2 for name in MODULE.CLASSES}
