from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_gpu_start_script_reports_the_first_missing_variable(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    for name in (
        "IDEAL_DATABASE_URL",
        "GPU_GATEWAY_SECRET",
        "SESSION_SECRET",
        "IDEAL_ADMIN_PASSWORD",
        "IDEAL_CONSENT_VERSION",
        "IDEAL_APP_VERSION",
        "IDEAL_STYLEGAN_REPO",
        "IDEAL_STYLEGAN_NETWORK",
        "IDEAL_PRIOR_PATH",
        "IDEAL_PERSONA_FEMALE_POOL",
        "IDEAL_PERSONA_MALE_POOL",
    ):
        environment.pop(name, None)
    environment["IDEAL_ENV_FILE"] = str(tmp_path / "absent.env")
    environment["IDEAL_PYTHON"] = sys.executable

    result = subprocess.run(
        ["bash", "scripts/run_gpu_backend.sh"],
        cwd=project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr.strip() == (
        "ERROR: IDEAL_DATABASE_URL is not configured."
    )
