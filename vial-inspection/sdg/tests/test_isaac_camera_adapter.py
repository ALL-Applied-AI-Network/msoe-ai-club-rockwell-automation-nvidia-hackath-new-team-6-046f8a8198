import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest


MODULE_PATH = Path(__file__).parents[1]/"isaac_camera_adapter.py"
SPEC = importlib.util.spec_from_file_location("isaac_camera_adapter", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_adapter_converts_and_deduplicates_frames(tmp_path):
    source = tmp_path/"render"
    incoming = tmp_path/"incoming"
    source.mkdir()
    incoming.mkdir()
    Image.new("RGB", (20, 10), "yellow").save(source/"rgb_0001.png")
    state = incoming/"state.json"

    first = MODULE.process_available(
        source, incoming, state, fixed_id="tray-A", expected_size=(20, 10))
    second = MODULE.process_available(
        source, incoming, state, fixed_id="tray-A", expected_size=(20, 10))

    assert [path.name for path in first] == ["tray-A__000001.jpg"]
    assert second == []
    assert Image.open(first[0]).size == (20, 10)
    assert len(json.loads(state.read_text())["processed"]) == 1


def test_adapter_rejects_wrong_dimensions(tmp_path):
    source = tmp_path/"frame.png"
    Image.new("RGB", (5, 5)).save(source)

    with pytest.raises(ValueError, match="expected 20x10"):
        MODULE.adapt_frame(source, tmp_path/"output.jpg", (20, 10))


def test_tray_id_validation_is_path_safe():
    assert MODULE.validate_tray_id("line-2_tray.7") == "line-2_tray.7"
    with pytest.raises(ValueError, match="tray ID"):
        MODULE.validate_tray_id("../escape")
