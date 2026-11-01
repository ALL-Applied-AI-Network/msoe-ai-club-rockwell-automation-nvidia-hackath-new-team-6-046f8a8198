#!/usr/bin/env python3
"""Atomically adapt Isaac RGB writer frames into inspection tray sessions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

from PIL import Image


STATE_SCHEMA = "vial-isaac-camera-adapter-1"
SOURCE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tray_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("tray ID must contain only letters, digits, dot, underscore, or dash")
    return value


def replace_with_retry(source, destination, attempts=10):
    delay = 0.01
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == attempts-1:
                raise
            time.sleep(delay)
            delay = min(delay*2, 0.25)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(path):
    path = Path(path)
    if not path.exists():
        return {"schema_version": STATE_SCHEMA, "processed": {}, "sequences": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != STATE_SCHEMA:
        raise ValueError("unsupported camera adapter state")
    return state


def source_frames(source, recursive=False):
    iterator = Path(source).rglob("*") if recursive else Path(source).iterdir()
    return sorted((path for path in iterator
                   if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES),
                  key=lambda path: path.as_posix().lower())


def current_tray_id(fixed_id, tray_id_file):
    if tray_id_file is None:
        return validate_tray_id(fixed_id)
    return validate_tray_id(Path(tray_id_file).read_text(encoding="utf-8").strip())


def adapt_frame(source, destination, expected_size=(1920, 1080)):
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with Image.open(source) as image:
            image.load()
            if image.size != expected_size:
                raise ValueError(
                    f"{Path(source).name}: expected {expected_size[0]}x{expected_size[1]}, "
                    f"got {image.size[0]}x{image.size[1]}")
            image.convert("RGB").save(temporary, format="JPEG", quality=95)
        replace_with_retry(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def process_available(source, incoming, state_path, fixed_id=None,
                      tray_id_file=None, recursive=False,
                      expected_size=(1920, 1080)):
    state = load_state(state_path)
    outcomes = []
    for path in source_frames(source, recursive):
        digest = sha256(path)
        key = f"{path.resolve()}:{digest}"
        if key in state["processed"]:
            continue
        tray_id = current_tray_id(fixed_id, tray_id_file)
        sequence = int(state["sequences"].get(tray_id, 0))+1
        output = Path(incoming)/f"{tray_id}__{sequence:06d}.jpg"
        adapt_frame(path, output, expected_size)
        state["sequences"][tray_id] = sequence
        state["processed"][key] = {
            "source": str(path.resolve()),
            "source_sha256": digest,
            "tray_id": tray_id,
            "output": output.name,
        }
        atomic_json(state_path, state)
        outcomes.append(output)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--incoming", required=True, type=Path)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--tray-id")
    identity.add_argument("--tray-id-file", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("source directory does not exist")
    args.incoming.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.incoming/"isaac-camera-adapter-state.json"
    if args.poll_seconds <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("poll interval and dimensions must be positive")
    while True:
        outputs = process_available(
            args.source, args.incoming, state_path, args.tray_id,
            args.tray_id_file, args.recursive, (args.width, args.height))
        for output in outputs:
            print(f"ADAPTED {output.name}", flush=True)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
