import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1]/"train_two_stage.py"
SPEC = importlib.util.spec_from_file_location("train_two_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_crop_path_supports_both_manifests():
    assert MODULE.crop_path({"_root": "/d", "image": "train/pass/a.png"}) == Path(
        "/d/train/pass/a.png")
    assert MODULE.crop_path({"_root": "/d", "image": "source.jpg",
                             "path": "train/pass/a.png"}) == Path("/d/train/pass/a.png")
