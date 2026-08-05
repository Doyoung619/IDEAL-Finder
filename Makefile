.PHONY: demo run test export models

demo:
	IDEAL_DEMO=1 python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

run:
	python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

test:
	python -m pytest

export:
	python scripts/export_data.py

models:
	python scripts/download_models.py --with-clip

