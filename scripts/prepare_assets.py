from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.runtime import ExperimentRuntime
from app.settings import load_config


def main() -> None:
    config = load_config()
    runtime = ExperimentRuntime(config)
    runtime.ensure_ready()
    latents = runtime.generator.sample_prior(2, config.experiment.seed)
    images = runtime.generator.decode(latents)
    genders = runtime.gender_controller.estimate(
        images,
        latents=latents,
        generator_name=runtime.generator.generator_name,
    )
    adults = runtime.adult_controller.estimate(
        images,
        generator_name=runtime.generator.generator_name,
    )
    output_path = Path(config.paths.output_dir) / "generator_preflight.png"
    images[0].save(output_path)
    print(
        "Filter preflight:",
        {
            "gender_backend": genders[0].backend,
            "gender": genders[0].label,
            "gender_confidence": round(genders[0].confidence, 4),
            "adult_backend": adults[0].backend,
            "adult": adults[0].is_adult,
            "adult_confidence": round(adults[0].confidence, 4),
        },
    )
    print(f"Generator, PCA, and decoder are ready: {output_path}")


if __name__ == "__main__":
    main()
