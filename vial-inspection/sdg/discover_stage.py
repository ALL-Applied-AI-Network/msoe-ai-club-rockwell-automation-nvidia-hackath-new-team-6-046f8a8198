"""Inventory vial-inspection USD prims before authoring synthetic overrides.

Run with Isaac Sim's Python. The script is read-only: it opens the supplied stage,
records hierarchy/semantics/transforms, and writes JSON outside the USD package.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--all-prims", action="store_true",
                        help="include every prim instead of the inspection-relevant subset")
    return parser.parse_args()


args = arguments()
stage_path = Path(args.stage).resolve()
output_path = Path(args.output).resolve()
if not stage_path.is_file():
    raise SystemExit(f"stage does not exist: {stage_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

try:
    import omni.usd
    from pxr import Usd, UsdGeom

    context = omni.usd.get_context()
    if not context.open_stage(str(stage_path)):
        raise RuntimeError(f"could not open stage: {stage_path}")
    while context.get_stage_loading_status()[2] > 0:
        app.update()
    for _ in range(10):
        app.update()
    stage = context.get_stage()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    terms = ("vial", "cap", "tray", "fixture", "aperture", "hole", "camera",
             "light", "robot", "fanuc", "scrap", "good", "inspection")
    records = []
    type_counts = Counter()
    total = 0
    for prim in stage.Traverse():
        total += 1
        type_name = prim.GetTypeName() or "untyped"
        type_counts[type_name] += 1
        path = str(prim.GetPath())
        name = prim.GetName()
        attrs = {}
        for attr in prim.GetAttributes():
            attr_name = attr.GetName()
            if ("semantic" in attr_name.lower() or "synthetic" in attr_name.lower()
                    or "class" in attr_name.lower()):
                value = attr.Get()
                if value is not None:
                    attrs[attr_name] = str(value)
        relevant = args.all_prims or any(term in (path+" "+name).lower() for term in terms) or bool(attrs)
        if not relevant:
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        imageable = UsdGeom.Imageable(prim)
        visibility = str(imageable.ComputeVisibility()) if imageable else None
        records.append({
            "path": path,
            "name": name,
            "type": type_name,
            "active": prim.IsActive(),
            "instance": prim.IsInstance(),
            "instanceable": prim.IsInstanceable(),
            "visibility": visibility,
            "world_translation_m": [round(float(v), 6) for v in translation],
            "semantic_attributes": attrs,
            "children": len(prim.GetChildren()),
        })
    payload = {
        "stage": str(stage_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "total_prims": total,
        "type_counts": dict(sorted(type_counts.items())),
        "included_prims": len(records),
        "prims": records,
    }
    output_path.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(f"DISCOVERY COMPLETE total={total} included={len(records)} output={output_path}", flush=True)
except BaseException:
    traceback.print_exc()
    app.close(exit_code=1)
    raise
app.close()
