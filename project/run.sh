#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python -m pip freeze > /job/out/environment.txt
cp train.py requirements.txt /job/out/
python -u train.py
