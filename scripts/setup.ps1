param([switch]$PrepareInitialCache)
$ErrorActionPreference = "Stop"

python -m pip install -r requirements.txt
python scripts\download_models.py --with-clip
python scripts\prepare_assets.py

if ($PrepareInitialCache) {
    python scripts\prepare_initial_cache.py
} else {
    Write-Host "Optional speed-up: run .\scripts\setup.ps1 -PrepareInitialCache"
    Write-Host "See INITIAL_RUNTIME_CACHE.md for details."
}

Write-Host "Setup complete. Run scripts\run_dev.ps1 to start the app."
