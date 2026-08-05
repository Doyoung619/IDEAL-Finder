$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
python scripts\download_models.py --with-clip
python scripts\prepare_assets.py

Write-Host "Setup complete. Run scripts\run_dev.ps1 to start the app."
