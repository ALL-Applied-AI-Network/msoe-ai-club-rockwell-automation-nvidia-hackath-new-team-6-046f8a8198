#!/usr/bin/env bash
set -euo pipefail
cp generate.py /job/out/
/isaac-sim/python.sh /job/in/generate.py --frames 500
test -s /job/out/manifest.json
tar -czf /job/out/dataset.tar.gz -C /job/out dataset manifest.json preview.jpg generate.py
sha256sum /job/out/dataset.tar.gz > /job/out/dataset.tar.gz.sha256
