"""Render a deterministic five-class proof scene from the CSI tray digital twin."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--camera-config", required=True)
    parser.add_argument("--apply-lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subframes", type=int, default=12)
    parser.add_argument("--crop-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--states-file", help="optional JSON mapping of G01-G30 to classes")
    parser.add_argument("--augment", action="store_true",
                        help="apply seeded defect-severity, lighting, and image-space camera jitter")
    return parser.parse_args()


args = parse_args()
stage_path = Path(args.stage).resolve()
camera_config_path = Path(args.camera_config).resolve()
apply_lens_path = Path(args.apply_lens).resolve()
output = Path(args.output).resolve()
for path in (stage_path, camera_config_path, apply_lens_path):
    if not path.is_file():
        raise SystemExit(f"required file does not exist: {path}")
output.mkdir(parents=True, exist_ok=True)
(output/"crops").mkdir(exist_ok=True)
camera_config = json.loads(camera_config_path.read_text())
rng = random.Random(args.seed)
if args.crop_size < 16 or args.crop_size % 2:
    raise SystemExit("--crop-size must be an even integer of at least 16 pixels")
render_width, render_height = camera_config["render_px"]

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting",
                     "width": render_width, "height": render_height})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


try:
    import cv2
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while context.get_stage_loading_status()[2] > 0:
        app.update()
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def prim(path):
        value = stage.GetPrimAtPath(path)
        if not value.IsValid():
            raise RuntimeError(f"required prim is missing: {path}")
        return value

    def hide(path):
        UsdGeom.Imageable(prim(path)).MakeInvisible()

    def rotate_and_translate(path, rotation, translation):
        xform = UsdGeom.Xformable(prim(path))
        xform.AddRotateXYZOp(opSuffix="sdg").Set(Gf.Vec3f(*rotation))
        xform.AddTranslateOp(opSuffix="sdg").Set(Gf.Vec3d(*translation))

    def deform(path, scale, rotation):
        xform = UsdGeom.Xformable(prim(path))
        xform.AddScaleOp(opSuffix="sdg").Set(Gf.Vec3f(*scale))
        xform.AddRotateXYZOp(opSuffix="sdg").Set(Gf.Vec3f(*rotation))

    def translate(path, value):
        UsdGeom.Xformable(prim(path)).AddTranslateOp(opSuffix="damage").Set(Gf.Vec3d(*value))

    # Target states. All geometry changes live only in the anonymous session layer.
    allowed_classes = {"pass", "missing_cap", "crooked_cap", "damaged_vial", "no_vial"}
    if args.states_file:
        classes = json.loads(Path(args.states_file).read_text())
        expected = {f"G{i:02}" for i in range(1, 31)}
        if set(classes) != expected or not set(classes.values()) <= allowed_classes:
            raise ValueError("states file must map exactly G01-G30 to supported classes")
    else:
        classes = {f"G{i:02}": "pass" for i in range(1, 31)}
        classes.update({"G02": "missing_cap", "G03": "crooked_cap",
                        "G04": "damaged_vial", "G05": "no_vial"})

    defect_parameters = {}
    for slot_id, state in classes.items():
        base = f"/TrayInspection/{slot_id}"
        if state == "missing_cap":
            hide(base+"_cap")
        elif state == "no_vial":
            hide(base+"_tube")
            hide(base+"_cap")
        elif state == "crooked_cap":
            severity = rng.uniform(0.75, 1.3) if args.augment else 1.0
            rotation = (28.0*severity, 18.0*severity, rng.uniform(-10.0, 10.0))
            translation = (0.014*severity, rng.uniform(-0.006, 0.006), -0.006*severity)
            rotate_and_translate(base+"_cap", rotation, translation)
            defect_parameters[slot_id] = {"severity": round(severity, 5)}
        elif state == "damaged_vial":
            severity = rng.uniform(0.75, 1.25) if args.augment else 1.0
            scale = (max(0.24, 0.38/severity), 1.0+0.55*severity, max(0.38, 0.58/severity))
            deform(base+"_tube", scale, (32.0*severity, 11.0, 18.0))
            translate(base+"_tube", (0.018*severity, -0.008, -0.012))
            rotate_and_translate(base+"_cap", (24.0*severity, 10.0, 18.0),
                                 (0.026*severity, -0.012, -0.025*severity))
            defect_parameters[slot_id] = {"severity": round(severity, 5)}

    # Make the scrap tray deterministically empty for this proof scene.
    for position in range(1, 31):
        classes[f"S{position:02}"] = "no_vial"
        for suffix in ("tube", "cap"):
            path = f"/TrayInspection/S{position:02}_{suffix}"
            value = stage.GetPrimAtPath(path)
            if value.IsValid():
                UsdGeom.Imageable(value).MakeInvisible()

    camera_path = camera_config["prim_in_station_file"]
    camera_prim = prim(camera_path)
    render_product = rep.create.render_product(camera_prim, (render_width, render_height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([render_product])
    rep.orchestrator.set_capture_on_play(False)

    # Force the scaled camera clipping range to re-sync, matching render_example.py.
    clip = UsdGeom.Camera(camera_prim).GetClippingRangeAttr()
    near_far = clip.Get()
    clip.Set(Gf.Vec2f(near_far[0]*0.5, near_far[1])); app.update()
    clip.Set(near_far); app.update()
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
    pixels = np.asarray(rgb.get_data())
    if pixels.ndim != 3 or pixels.shape[:2] != (render_height, render_width) or pixels.std() < 3:
        raise RuntimeError(f"invalid RGB capture: shape={pixels.shape} std={pixels.std():.3f}")
    pinhole = output/"proof_scene.pinhole.png"
    if not cv2.imwrite(str(pinhole), pixels[..., :3][..., ::-1]):
        raise OSError(f"could not write {pinhole}")
    final_image = output/"proof_scene.png"
    subprocess.run([sys.executable, str(apply_lens_path), str(pinhole),
                    str(camera_config_path), str(final_image)], check=True)
    pinhole.unlink()

    # Project fixed world targets through the supplied calibrated distorted camera.
    camera_position = np.asarray(camera_config["position_m"], dtype=float)
    rotation = np.asarray(camera_config["R_world_to_cv"], dtype=float)
    focal = float(camera_config["focal_px_final"])
    cx, cy = camera_config["principal_point_final_px"]
    k1 = float(camera_config["lens_k1"])

    def project(world):
        x, y, z = rotation @ (np.asarray(world, dtype=float)-camera_position)
        if z <= 0:
            raise RuntimeError(f"target is behind camera: {world}")
        xn, yn = x/z, y/z
        factor = 1.0+k1*(xn*xn+yn*yn)
        return [float(focal*xn*factor+cx), float(focal*yn*factor+cy)]

    targets = {}
    good_world = {}
    for position in range(1, 31):
        path = f"/TrayInspection/G{position:02}_cap"
        world = cache.GetLocalToWorldTransform(prim(path)).ExtractTranslation()
        good_world[position] = [float(v) for v in world]
        targets[f"G{position:02}"] = project(good_world[position])

    # The packaged stage has one modeled scrap column. Extrapolate the remaining
    # fixed apertures with the measured good-tray column pitch; rows use modeled S01-S05.
    good_column_pitch = float(np.mean([good_world[p+5][0]-good_world[p][0] for p in range(1, 26)]))
    scrap_first_column = []
    for position in range(1, 6):
        path = f"/TrayInspection/S{position:02}_tube"
        world = cache.GetLocalToWorldTransform(prim(path)).ExtractTranslation()
        scrap_first_column.append([float(v) for v in world])
    for position in range(1, 31):
        column, row = divmod(position-1, 5)
        world = list(scrap_first_column[row])
        world[0] += column*good_column_pitch
        targets[f"S{position:02}"] = project(world)

    final = cv2.imread(str(final_image), cv2.IMREAD_COLOR)
    if final is None or final.shape[:2] != (camera_config["final_image_px"][1],
                                           camera_config["final_image_px"][0]):
        raise RuntimeError("lens output has the wrong dimensions")
    augmentation = {"brightness_gain": 1.0, "noise_sigma": 0.0,
                    "camera_shift_xy_px": [0, 0]}
    if args.augment:
        gain = rng.uniform(0.82, 1.18)
        noise_sigma = rng.uniform(0.0, 3.0)
        dx, dy = rng.randint(-4, 4), rng.randint(-4, 4)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        final = cv2.warpAffine(final, matrix, (final.shape[1], final.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        noise = np.random.default_rng(args.seed).normal(0.0, noise_sigma, final.shape)
        final = np.clip(final.astype(np.float32)*gain+noise, 0, 255).astype(np.uint8)
        if not cv2.imwrite(str(final_image), final):
            raise OSError(f"could not write augmented image: {final_image}")
        targets = {key: [value[0]+dx, value[1]+dy] for key, value in targets.items()}
        augmentation = {"brightness_gain": round(gain, 5),
                        "noise_sigma": round(noise_sigma, 5),
                        "camera_shift_xy_px": [dx, dy]}
    crop_size = args.crop_size
    half = crop_size//2
    slots = []
    for slot_id in [f"G{i:02}" for i in range(1, 31)]+[f"S{i:02}" for i in range(1, 31)]:
        u, v = targets[slot_id]
        left, top = int(round(u))-half, int(round(v))-half
        right, bottom = left+crop_size, top+crop_size
        if not (0 <= left < right <= final.shape[1] and 0 <= top < bottom <= final.shape[0]):
            raise RuntimeError(f"projected crop outside image: {slot_id} {(u, v)}")
        crop_path = output/"crops"/(slot_id+".png")
        cv2.imwrite(str(crop_path), final[top:bottom, left:right])
        slots.append({"id": slot_id, "class": classes[slot_id],
                      "center_xy": [round(u, 3), round(v, 3)],
                      "bbox_xyxy": [left, top, right, bottom],
                      "crop": str(crop_path.relative_to(output)).replace(os.sep, "/")})

    labels = {"schema_version": "sdg-scene-1", "image": "proof_scene.png",
              "stage": str(stage_path), "camera": camera_config["camera"],
              "seed": args.seed, "augmentation": augmentation, "slots": slots}
    (output/"labels.json").write_text(json.dumps(labels, indent=2)+"\n")
    files = {str(path.relative_to(output)).replace(os.sep, "/"): sha256(path)
             for path in output.rglob("*") if path.is_file()}
    manifest = {"seed": args.seed, "deterministic_states": classes,
                "defect_parameters": defect_parameters, "augmentation": augmentation,
                "source_stage_sha256": sha256(stage_path), "files": files}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"PROOF SCENE VERIFIED image={final_image} crops={len(slots)}", flush=True)
    rgb.detach([render_product])
    render_product.destroy()
except BaseException:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()
