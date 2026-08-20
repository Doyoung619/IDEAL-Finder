from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "outputs" / "algorithm_audit"
PAIR_KEY = (
    "seed",
    "persona_quality",
    "m_value",
    "algorithm",
    "noise",
    "mismatch",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in PAIR_KEY)


def paired_metrics(
    candidate: list[dict[str, str]], baseline: dict[tuple[str, ...], dict[str, str]]
) -> dict[str, float | int]:
    differences = [
        float(row["final_error"]) - float(baseline[key(row)]["final_error"])
        for row in candidate
    ]
    mean_difference = statistics.mean(differences)
    standard_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    half_width = 1.96 * standard_deviation / math.sqrt(len(differences))
    baseline_error = statistics.mean(
        float(baseline[key(row)]["final_error"]) for row in candidate
    )
    candidate_error = statistics.mean(float(row["final_error"]) for row in candidate)
    recovery_difference = statistics.mean(
        (row["recovered"] == "True")
        - (baseline[key(row)]["recovered"] == "True")
        for row in candidate
    )
    catastrophic_difference = statistics.mean(
        (row["catastrophic_failure"] == "True")
        - (baseline[key(row)]["catastrophic_failure"] == "True")
        for row in candidate
    )
    baseline_latency = statistics.mean(
        float(baseline[key(row)]["mean_optimization_latency_ms"])
        for row in candidate
    )
    candidate_latency = statistics.mean(
        float(row["mean_optimization_latency_ms"]) for row in candidate
    )
    return {
        "n": len(candidate),
        "baseline_error_mean": baseline_error,
        "candidate_error_mean": candidate_error,
        "paired_error_delta_mean": mean_difference,
        "paired_error_delta_ci95_low": mean_difference - half_width,
        "paired_error_delta_ci95_high": mean_difference + half_width,
        "relative_error_change": candidate_error / baseline_error - 1.0,
        "recovery_rate_change": recovery_difference,
        "catastrophic_rate_change": catastrophic_difference,
        "baseline_optimization_latency_ms": baseline_latency,
        "candidate_optimization_latency_ms": candidate_latency,
        "relative_latency_change": candidate_latency / baseline_latency - 1.0,
    }


def mean_metrics(rows: list[dict[str, str]]) -> dict[str, float | int]:
    return {
        "n": len(rows),
        "initial_error_mean": statistics.mean(float(row["initial_error"]) for row in rows),
        "final_error_mean": statistics.mean(float(row["final_error"]) for row in rows),
        "final_error_median": statistics.median(float(row["final_error"]) for row in rows),
        "recovery_rate": statistics.mean(row["recovered"] == "True" for row in rows),
        "catastrophic_failure_rate": statistics.mean(
            row["catastrophic_failure"] == "True" for row in rows
        ),
        "global_support_extinct_rate": statistics.mean(
            row["global_support_extinct"] == "True" for row in rows
        ),
        "final_global_mass_mean": statistics.mean(
            float(row["final_global_mass"]) for row in rows
        ),
        "optimization_latency_ms_mean": statistics.mean(
            float(row["mean_optimization_latency_ms"]) for row in rows
        ),
    }


def main() -> None:
    baseline_rows = read_csv(OUTPUT / "stress_test_runs.csv")
    fixed_rows = read_csv(OUTPUT / "fixed_beta" / "stress_test_runs.csv")
    mismatch_rows = read_csv(OUTPUT / "model_mismatch" / "stress_test_runs.csv")
    exploration_rows = read_csv(
        OUTPUT / "exploration_dev" / "stress_test_runs.csv"
    )
    confirmatory_rows = read_csv(
        OUTPUT / "confirmatory" / "stress_test_runs.csv"
    )
    baseline = {key(row): row for row in baseline_rows}

    beta_rows: list[dict[str, object]] = []
    for algorithm in ("entropy", "rc_mlq"):
        for persona in ("aligned", "moderate", "strong"):
            selected = [
                row
                for row in fixed_rows
                if row["algorithm"] == algorithm
                and row["persona_quality"] == persona
            ]
            metrics = paired_metrics(selected, baseline)
            # paired_metrics computes candidate-baseline; candidate here is fixed beta.
            beta_rows.append(
                {
                    "algorithm": algorithm,
                    "persona_quality": persona,
                    "comparison": "fixed_minus_adaptive",
                    **metrics,
                    "recommendation": "RECOMMEND FIXED FOR CLEANER EXPERIMENT",
                }
            )
    write_csv(OUTPUT / "beta_ablation.csv", beta_rows)

    development_baseline = {
        key(row): row
        for row in baseline_rows
        if int(row["seed"]) < 50
        and row["persona_quality"] in {"aligned", "strong"}
    }
    exploration_summary: list[dict[str, object]] = []
    for rho in (0.025, 0.05, 0.10, 0.20):
        rho_text = str(rho)
        for algorithm in ("entropy", "rc_mlq"):
            for persona in ("aligned", "strong"):
                selected = [
                    row
                    for row in exploration_rows
                    if float(row["exploration_rho"]) == rho
                    and row["algorithm"] == algorithm
                    and row["persona_quality"] == persona
                ]
                exploration_summary.append(
                    {
                        "split": "development_seeds_0_49",
                        "rho": rho_text,
                        "algorithm": algorithm,
                        "persona_quality": persona,
                        **paired_metrics(selected, development_baseline),
                        "selected_for_heldout": False,
                        "recommendation": "REJECT",
                    }
                )
    write_csv(OUTPUT / "exploration_ablation.csv", exploration_summary)

    all_mismatch = baseline_rows + mismatch_rows
    mismatch_summary: list[dict[str, object]] = []
    for algorithm in ("entropy", "rc_mlq"):
        for mismatch in ("under", "match", "over"):
            selected = [
                row
                for row in all_mismatch
                if row["algorithm"] == algorithm and row["mismatch"] == mismatch
            ]
            mismatch_summary.append(
                {
                    "algorithm": algorithm,
                    "mismatch": mismatch,
                    **mean_metrics(selected),
                    "final_beta_mean": statistics.mean(
                        float(row["final_beta_model"]) for row in selected
                    ),
                }
            )
    write_csv(OUTPUT / "model_mismatch.csv", mismatch_summary)

    runtime_rows: list[dict[str, object]] = []
    for profile_name, source in (
        ("audit_reduced", baseline_rows),
        ("confirmatory_higher_compute", confirmatory_rows),
    ):
        for algorithm in ("entropy", "rc_mlq"):
            values = [
                float(row["mean_optimization_latency_ms"])
                for row in source
                if row["algorithm"] == algorithm
            ]
            runtime_rows.append(
                {
                    "profile": profile_name,
                    "algorithm": algorithm,
                    "n_trajectories": len(values),
                    "optimization_latency_ms_mean": statistics.mean(values),
                    "optimization_latency_ms_median": statistics.median(values),
                    "optimization_latency_ms_std": statistics.stdev(values),
                    "optimization_latency_ms_p95": sorted(values)[
                        min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
                    ],
                    "render_latency_ms": "not_run_missing_stylegan_artifacts",
                    "total_gpu_query_latency_ms": "not_run_missing_stylegan_artifacts",
                }
            )
    write_csv(OUTPUT / "runtime_summary.csv", runtime_rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    confirm_values = {
        algorithm: [
            float(row["mean_optimization_latency_ms"])
            for row in confirmatory_rows
            if row["algorithm"] == algorithm
        ]
        for algorithm in ("entropy", "rc_mlq")
    }
    means = [statistics.mean(confirm_values[name]) for name in confirm_values]
    errors = [
        1.96 * statistics.stdev(confirm_values[name]) / math.sqrt(len(confirm_values[name]))
        for name in confirm_values
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    bars = axis.bar(
        ("Entropy", "RC-MLQ"),
        means,
        yerr=errors,
        capsize=5,
        color=("#276FBF", "#D1495B"),
    )
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errors) + 2,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
        )
    axis.set_ylim(0, max(means) * 1.25)
    axis.set_ylabel("Optimization latency per query (ms)")
    axis.set_title("Higher-compute confirmatory profile on CPU")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "runtime.pdf")
    plt.close(figure)

    mock_path = OUTPUT.parent / "query_diversity_diagnostic_mock" / "metrics.json"
    mock_rows = json.loads(mock_path.read_text(encoding="utf-8"))
    diversity_rows: list[dict[str, object]] = []
    for row in mock_rows:
        diversity_rows.append(
            {
                "source": "mock_generator_mock_embedding",
                "algorithm": row["algorithm"],
                "stage": row["uncertainty"],
                "guard_mode": row["mode"],
                "guard_triggered": row["diversity_guard_triggered"],
                "first_trigger_round": "",
                "max_similarity_before": row[
                    "pre_guard_max_pairwise_image_similarity"
                ],
                "max_similarity_after": row["max_pairwise_image_similarity"],
                "mean_similarity_after": row["mean_pairwise_image_similarity"],
                "min_pairwise_theta_distance": row[
                    "min_pairwise_theta_distance"
                ],
                "information_gain_before": row.get(
                    "information_gain_before_correction", ""
                ),
                "information_gain_after": row.get(
                    "information_gain_after_correction", ""
                ),
                "relative_information_gain_loss": row.get(
                    "relative_information_gain_loss", ""
                ),
                "radius_expansion_factor": row["radius_expansion_factor"],
                "orthogonal_fallback_used": row["orthogonal_fallback_used"],
                "interpretation": (
                    "diagnostic_only_threshold_null_no_intervention"
                ),
            }
        )
    write_csv(OUTPUT / "diversity_ablation.csv", diversity_rows)

    implementation_rows = [
        ("Persona warm start", "implemented", "Selected persona theta initializes every block"),
        ("Persona/global mixture prior", "implemented", "Exact 0.55 local + 0.45 global particle weights"),
        ("Particle posterior", "implemented", "Weighted particles with M-way likelihood update"),
        ("Entropy Query", "implemented", "Torch direct MI optimizer; slim NumPy fallback is approximate"),
        ("RC-MLQ", "implemented", "Principal-axis quantiles and exact EIG physical grid"),
        ("Adaptive beta", "implemented", "Bounded log-space update; optional flag"),
        ("Liu-West rejuvenation", "implemented", "Deterministic regularized resampling below ESS threshold"),
        ("Query diversity guard", "partially implemented", "Guard exists; real threshold null and latent_min_distance unused"),
        ("East-Asian conditional prior", "partially implemented", "Config/build/validate path exists; local artifacts absent"),
        ("Query-time global exploration", "not implemented", "Development candidate failed and was removed from production"),
        ("Predictive-surprise / soft reopen", "not implemented", "No execution path found"),
        ("final_query_points consistency", "implemented", "Post-guard displayed theta is persisted"),
        ("Fixed-beta mode", "implemented", "adaptive_beta_enabled=false"),
        ("Query diversity disable flag", "implemented", "enabled=false is diagnostic-only and preserves theta"),
    ]
    write_csv(
        OUTPUT / "implementation_audit.csv",
        [
            {"feature": feature, "status": status, "evidence": evidence}
            for feature, status, evidence in implementation_rows
        ],
    )

    baseline_by_persona_algorithm: dict[str, object] = {}
    for persona in ("aligned", "moderate", "strong"):
        for algorithm in ("entropy", "rc_mlq"):
            selected = [
                row
                for row in baseline_rows
                if row["persona_quality"] == persona
                and row["algorithm"] == algorithm
            ]
            baseline_by_persona_algorithm[f"{persona}:{algorithm}"] = mean_metrics(
                selected
            )
    confirmatory: dict[str, object] = {}
    for persona in ("aligned", "strong"):
        for algorithm in ("entropy", "rc_mlq"):
            selected = [
                row
                for row in confirmatory_rows
                if row["persona_quality"] == persona
                and row["algorithm"] == algorithm
            ]
            confirmatory[f"{persona}:{algorithm}"] = mean_metrics(selected)
    report = {
        "baseline_by_persona_algorithm": baseline_by_persona_algorithm,
        "confirmatory_high_compute": confirmatory,
        "decisions": {
            "query_time_global_exploration": {
                "recommendation": "REJECT",
                "selected_rho": None,
                "heldout_evaluated": False,
                "reason": "No development rho met the predeclared strong-wrong improvement criterion.",
            },
            "adaptive_beta": {
                "recommendation": "RECOMMEND FIXED FOR CLEANER EXPERIMENT",
                "user_study_default": False,
            },
            "diversity_guard": {
                "recommendation": "KEEP OPTIONAL",
                "user_study_effective_state": "diagnostic_only_until_real_threshold_is_validated",
                "real_stylegan_validation": "not_run_missing_artifacts",
            },
            "early_stopping": {
                "recommendation": "DIAGNOSTIC ONLY",
                "experiment_round_count_changed": False,
            },
        },
        "artifact_availability": {
            "stylegan_checkpoint": False,
            "east_asian_prior": False,
            "fairface_weights": False,
            "real_stylegan_openclip_validation": "not_run",
        },
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("Wrote audit ablations and analysis summary to", OUTPUT)


if __name__ == "__main__":
    main()
