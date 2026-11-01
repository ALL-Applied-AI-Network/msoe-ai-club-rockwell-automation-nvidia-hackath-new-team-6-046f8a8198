"""Generate a reproducible, scene-split vial inspection dataset with Isaac Sim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys


CLASSES = ("pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--apply-lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--subframes", type=int, default=12)
    return parser.parse_args()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def balanced_states(seed):
    values = [name for name in CLASSES for _ in range(6)]
    random.Random(seed).shuffle(values)
    return {f"G{index:02}": value for index, value in enumerate(values, 1)}


def main():
    args = parse_args()
    if args.scenes < 2:
        raise SystemExit("--scenes must be at least 2")
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_ids = [f"scene_{index:05}" for index in range(args.scenes)]
    shuffled = list(scene_ids)
    random.Random(args.seed).shuffle(shuffled)
    val_count = min(args.scenes-1, max(1, round(args.scenes*args.val_fraction)))
    val_ids = set(shuffled[:val_count])
    generator = Path(__file__).with_name("generate_proof_scene.py")
    records = []
    totals = Counter()

    for index, scene_id in enumerate(scene_ids):
        scene_seed = args.seed+index
        scene_dir = output/scene_id
        scene_dir.mkdir(exist_ok=True)
        states = balanced_states(scene_seed)
        states_path = scene_dir/"states.json"
        states_path.write_text(json.dumps(states, indent=2)+"\n")
        command = [sys.executable, str(generator), "--stage", args.stage,
                   "--camera-config", args.camera_config, "--apply-lens", args.apply_lens,
                   "--output", str(scene_dir), "--seed", str(scene_seed),
                   "--states-file", str(states_path), "--augment",
                   "--crop-size", str(args.crop_size), "--subframes", str(args.subframes)]
        print(f"[{index+1}/{args.scenes}] {scene_id} seed={scene_seed}", flush=True)
        subprocess.run(command, check=True)
        labels_path = scene_dir/"labels.json"
        labels = json.loads(labels_path.read_text())
        counts = Counter(slot["class"] for slot in labels["slots"])
        totals.update(counts)
        records.append({"id": scene_id, "seed": scene_seed,
                        "split": "val" if scene_id in val_ids else "train",
                        "labels": str(labels_path.relative_to(output)).replace("\\", "/"),
                        "class_counts": dict(sorted(counts.items())),
                        "labels_sha256": sha256(labels_path)})

    dataset = {"schema_version": "sdg-dataset-1", "seed": args.seed,
               "scene_count": args.scenes,
               "splits": {"train": args.scenes-val_count, "val": val_count},
               "good_tray_policy": "exactly six examples per class per scene",
               "scrap_tray_policy": "30 no_vial positions per scene",
               "class_counts": dict(sorted(totals.items())), "scenes": records}
    (output/"dataset.json").write_text(json.dumps(dataset, indent=2)+"\n")
    print(f"DATASET VERIFIED scenes={args.scenes} train={args.scenes-val_count} "
          f"val={val_count} slots={sum(totals.values())}", flush=True)


if __name__ == "__main__":
    main()
