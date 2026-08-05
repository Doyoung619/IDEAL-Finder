#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt
python scripts/download_models.py --with-clip
python scripts/prepare_assets.py

if [[ "${PREPARE_INITIAL_CACHE:-0}" == "1" ]]; then
  python scripts/prepare_initial_cache.py
else
  echo "Optional speed-up: PREPARE_INITIAL_CACHE=1 bash scripts/setup.sh"
  echo "See INITIAL_RUNTIME_CACHE.md for details."
fi

echo "Setup complete. Run scripts/run_dev.sh to start the app."
