#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python scripts/download_models.py --with-clip
python scripts/prepare_assets.py

echo "Setup complete. Run scripts/run_dev.sh to start the app."
