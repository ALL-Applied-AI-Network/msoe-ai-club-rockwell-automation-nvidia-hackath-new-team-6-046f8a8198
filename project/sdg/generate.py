import argparse
import hashlib
import json
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--frames', type=int, default=500)
parser.add_argument('--output', default='/job/out/dataset')
args, _ = parser.parse_known_args()
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting', 'width':640, 'height':480})
import carb
import numpy as np
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom, UsdLux, Gf
from isaacsim.core.utils.semantics import add_labels
from PIL import Image, ImageDraw

out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
try:
    rng = np.random.default_rng(42)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set('/rtx/post/dlss/execMode', 2)
    table = UsdGeom.Cube.Define(stage, '/World/Table')
    table.CreateSizeAttr(1)
    table.CreateDisplayColorAttr([(0.25,0.27,0.30)])
    tx = UsdGeom.Xformable(table)
    tx.AddTranslateOp().Set(Gf.Vec3d(0,0,-0.04))
    tx.AddScaleOp().Set(Gf.Vec3f(1.4,1.1,0.08))
    colors = [('yellow',(0.95,0.75,0.03)),('blue',(0.03,0.15,0.9)),('red',(0.9,0.04,0.03))]
    objects=[]
    for i in range(6):
        name, color = colors[i//2]
        cube = UsdGeom.Cube.Define(stage, f'/World/Cube_{i}')
        cube.CreateSizeAttr(0.09)
        cube.CreateDisplayColorAttr([color])
        add_labels(cube.GetPrim(), labels=[name+'_cube'], instance_name='class')
        x = UsdGeom.Xformable(cube)
        objects.append((x.AddTranslateOp(), x.AddRotateZOp(), name))
    dome = UsdLux.DomeLight.Define(stage, '/World/Dome')
    dome.CreateIntensityAttr(350)
    sun = UsdLux.DistantLight.Define(stage, '/World/Sun')
    sun.CreateIntensityAttr(1800)
    sunrot = UsdGeom.Xformable(sun).AddRotateXYZOp()
    camera = UsdGeom.Camera.Define(stage, '/World/Camera')
    camera.CreateFocalLengthAttr(30)
    camera.CreateHorizontalApertureAttr(36)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01,100))
    camop = UsdGeom.Xformable(camera).AddTransformOp()
    camop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(0,-1.0,1.3), Gf.Vec3d(0,0,0), Gf.Vec3d(0,0,1)).GetInverse())
    product = rep.create.render_product('/World/Camera', (640,480))
    for _ in range(30): app.update()
    # Initialize render/annotation pipelines before recording numbered frames.
    rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
    writer = rep.writers.get('BasicWriter')
    writer.initialize(output_dir=str(out), rgb=True, bounding_box_2d_tight=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(product)
    metadata=[]
    for frame in range(args.frames):
        poses=[]
        for i,(pos,rot,name) in enumerate(objects):
            xyz = [float((i%3-1)*0.25+rng.uniform(-0.04,0.04)), float((i//3-0.5)*0.3+rng.uniform(-0.04,0.04)),0.045]
            yaw=float(rng.uniform(0,360))
            pos.Set(Gf.Vec3d(*xyz)); rot.Set(yaw)
            poses.append({'class':name+'_cube','position':xyz,'yaw_deg':yaw})
        eye = Gf.Vec3d(float(rng.uniform(-0.22,0.22)),float(rng.uniform(-1.1,-0.85)),float(rng.uniform(1.1,1.45)))
        camop.Set(Gf.Matrix4d().SetLookAt(eye,Gf.Vec3d(0,0,0),Gf.Vec3d(0,0,1)).GetInverse())
        intensity=float(rng.uniform(900,2400))
        sun.GetIntensityAttr().Set(intensity)
        sunrot.Set(Gf.Vec3f(float(rng.uniform(-50,-20)),float(rng.uniform(-30,30)),float(rng.uniform(0,360))))
        dome.GetIntensityAttr().Set(float(rng.uniform(200,500)))
        table.GetDisplayColorAttr().Set([(float(rng.uniform(.15,.45)),)*3])
        rep.orchestrator.step(rt_subframes=8, delta_time=0.0)
        metadata.append({'frame':frame,'objects':poses,'camera_position':list(eye),'sun_intensity':intensity})
        if frame%25==0: print(f'CAPTURE {frame+1}/{args.frames}',flush=True)
    rep.orchestrator.wait_until_complete()
    writer.detach()
    product.destroy()
    (out/'scene_metadata.json').write_text(json.dumps(metadata,indent=2))
    rgb=sorted(out.glob('rgb_*.png'))
    boxes=sorted(out.glob('bounding_box_2d_tight_*.npy'))
    assert len(rgb)==args.frames, f'Expected {args.frames} RGB images; got {len(rgb)}'
    assert len(boxes)==args.frames, f'Expected {args.frames} box files; got {len(boxes)}'
    hashes=[]
    for p,b in zip(rgb,boxes):
        a=np.asarray(Image.open(p).convert('RGB'))
        assert a.shape==(480,640,3) and a.std()>3, f'Invalid or blank image: {p}'
        labels=np.load(b,allow_pickle=False)
        assert len(labels)>=6, f'Missing objects in {b}: {len(labels)}'
        hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
    assert len(set(hashes))==args.frames,'Duplicate RGB images'
    sheet=Image.new('RGB',(960,720))
    for j,index in enumerate(np.linspace(0,args.frames-1,9,dtype=int)):
        tile=Image.open(rgb[index]).convert('RGB').resize((320,240))
        ImageDraw.Draw(tile).text((8,8),str(index),fill='white')
        sheet.paste(tile,((j%3)*320,(j//3)*240))
    sheet.save(out.parent/'preview.jpg')
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out.parent/'manifest.json').write_text(json.dumps({'frames':args.frames,'seed':42,'resolution':[640,480],'scene':'procedural tabletop colored cubes; not calibrated FANUC cell','rgb_unique':len(set(hashes)),'files':files},indent=2))
    print('DATASET VERIFIED',len(rgb),flush=True)
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
