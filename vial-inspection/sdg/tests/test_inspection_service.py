import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1]/"inspection_service.py"
SPEC = importlib.util.spec_from_file_location("inspection_service", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_discover_images_is_sorted_and_filters_non_images(tmp_path):
    (tmp_path/"b.JPG").write_bytes(b"b")
    (tmp_path/"A.jpeg").write_bytes(b"a")
    (tmp_path/"ignore.png").write_bytes(b"png")
    (tmp_path/"notes.txt").write_text("notes")

    names = [path.name for path in MODULE.discover_images(tmp_path)]

    assert names == ["A.jpeg", "b.JPG"]


def test_atomic_json_replaces_complete_document(tmp_path):
    target = tmp_path/"state.json"
    MODULE.atomic_write_json(target, {"value": 1})
    MODULE.atomic_write_json(target, {"value": 2, "ready": True})

    assert json.loads(target.read_text()) == {"value": 2, "ready": True}
    assert not list(tmp_path.glob(".*.tmp"))


def test_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    source = tmp_path/"source"
    destination = tmp_path/"destination"
    source.write_text("new")
    destination.write_text("old")
    real_replace = MODULE.os.replace
    attempts = []

    def flaky_replace(first, second):
        attempts.append((first, second))
        if len(attempts) < 3:
            raise PermissionError("sharing violation")
        real_replace(first, second)

    monkeypatch.setattr(MODULE.os, "replace", flaky_replace)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _delay: None)

    MODULE.replace_with_retry(source, destination)

    assert destination.read_text() == "new"
    assert len(attempts) == 3


def test_state_rejects_bundle_change(tmp_path):
    state_path = tmp_path/"state.json"
    MODULE.atomic_write_json(state_path, MODULE.fresh_state("first"))

    with pytest.raises(ValueError, match="different bundle"):
        MODULE.load_state(state_path, "second")


def test_route_actions_are_fail_safe():
    assert MODULE.route_action("complete_order") == "accept"
    assert MODULE.route_action("incomplete_order") == "reject"
    assert MODULE.route_action("unresolved") == "hold"
    with pytest.raises(ValueError, match="unsupported tray verdict"):
        MODULE.route_action("unknown")
