$ErrorActionPreference = "Stop"
Remove-Item Env:IDEAL_DEMO -ErrorAction SilentlyContinue
python scripts\invalidate_legacy_sessions.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
