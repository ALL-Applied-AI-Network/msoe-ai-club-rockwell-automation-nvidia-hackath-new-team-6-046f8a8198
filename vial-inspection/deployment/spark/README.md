# Spark inspection runtime

The Spark workspace is mounted at `/workspace` inside the `isaac` container.
The launcher uses Isaac Sim's bundled Python, so it does not install packages or
modify the container image.

## Start and inspect the runtime

From the Spark host:

```bash
/home/yt-hamstersc2e8/workspace/vial-inspection/deployment/spark/inspection_runtime.sh start
/home/yt-hamstersc2e8/workspace/vial-inspection/deployment/spark/inspection_runtime.sh status
```

The watcher reads `/workspace/vial-inspection/live/incoming` and writes results
to `/workspace/vial-inspection/live/results`. The report server binds only to
the Spark's loopback interface on port 8765.

From the Windows workstation, keep this tunnel open:

```powershell
ssh -N -L 8765:127.0.0.1:8765 yt-hamstersc2e8@172.30.102.122
```

Then open <http://127.0.0.1:8765/report.html>. Stop or restart the remote
runtime with the corresponding `stop` or `restart` launcher action.

## Adapt Isaac camera frames

Configure the Isaac RGB writer to emit 1920x1080 PNG or JPEG frames into a
source directory. Start the adapter inside the container with either a fixed
tray ID or a controller-managed tray-ID file:

```bash
docker exec -d isaac /isaac-sim/python.sh \
  /workspace/vial-inspection/sdg/isaac_camera_adapter.py \
  --source /workspace/vial-inspection/live/isaac-rgb \
  --incoming /workspace/vial-inspection/live/incoming \
  --tray-id-file /workspace/vial-inspection/live/current-tray-id.txt
```

The tray-ID file must contain a path-safe ID such as `line2-tray-0042`. Change
it before the next physical tray begins. The adapter validates dimensions,
converts each completed writer frame to JPEG, assigns the session filename
`<tray-id>__<sequence>.jpg`, commits it atomically, and records source hashes so
restarts do not duplicate frames.

The runtime routes a tray only after the session policy is satisfied:

- `accept`: at least three frames, all base predictions complete-like, and at
  least one frame clears the locked confidence gate;
- `reject`: any frame is confidently incomplete;
- `hold`: insufficient or unresolved evidence.

Use a new tray ID after an accept or reject decision finalizes the prior ID.
