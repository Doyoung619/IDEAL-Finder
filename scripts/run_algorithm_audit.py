from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Keep each worker single-threaded. Parallelism is controlled at the trajectory level.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.entropy_query import EntropyQueryConfig, EntropyQuerySelector
from core.preference_posterior import ParticleMixturePreferencePosterior
from core.query_diversity import expected_information_gain
from core.rc_mlq import RCMLQConfig, RCMLQSelector
from scripts.query_exploration_candidate import acquisition_posterior


BASELINE_FILES = (
    "core/preference_posterior.py",
    "core/query_strategy.py",
    "core/entropy_query.py",
    "core/rc_mlq.py",
    "core/query_diversity.py",
    "core/conditional_prior.py",
    "app/services/runtime.py",
    "app/services/generation_service.py",
    "configs/default.yaml",
    "configs/persona_study.yaml",
)
PERSONA_DISTANCE = {"aligned": 0.5, "moderate": 2.5, "strong": 5.0}
USER_BETA = {"low": 2.5, "medium": 1.0, "high": 0.35}
MODEL_MULTIPLIER = {"match": 1.0, "over": 2.0, "under": 0.5}


@dataclass(frozen=True)
class ComputeProfile:
    seeds: int
    rounds: int
    particle_count: int
    entropy_mc_samples: int
    entropy_restarts: int
    entropy_steps: int
    rc_samples: int
    rc_steps: int


PROFILES = {
    "quick": ComputeProfile(3, 4, 512, 32, 1, 3, 512, 20),
    "audit": ComputeProfile(100, 12, 512, 64, 1, 8, 512, 60),
    "confirm": ComputeProfile(20, 12, 1024, 256, 3, 60, 2048, 120),
}


@dataclass(frozen=True)
class Task:
    seed: int
    persona_quality: str
    m_value: int
    algorithm: str
    noise: str
    mismatch: str
    adaptive_beta: bool
    exploration_rho: float
    profile: ComputeProfile


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def baseline_fingerprint() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    hashes: dict[str, str] = {}
    combined = hashlib.sha256()
    for relative in BASELINE_FILES:
        payload = (PROJECT_ROOT / relative).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        hashes[relative] = digest
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "commit": commit,
        "algorithm_version": "rc_mlq_v4_randomized_entropy_spread14",
        "algorithm_manifest_sha256": combined.hexdigest(),
        "files": hashes,
    }


def _scenario(task: Task) -> tuple[np.ndarray, np.ndarray]:
    random = np.random.default_rng(stable_seed("scenario", task.seed))
    target = random.normal(size=12)
    direction = random.normal(size=12)
    direction -= float(direction @ target) * target / max(float(target @ target), 1e-12)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-8:
        direction = np.roll(target, 1)
        norm = float(np.linalg.norm(direction))
    direction /= norm
    persona = target + PERSONA_DISTANCE[task.persona_quality] * direction
    return target, persona


def _selector(task: Task, beta: float):
    if task.algorithm == "entropy":
        return EntropyQuerySelector(
            EntropyQueryConfig(
                posterior_mc_samples=task.profile.entropy_mc_samples,
                num_restarts=task.profile.entropy_restarts,
                optimization_steps=task.profile.entropy_steps,
                learning_rate=0.05,
                seed=0,
                device="cpu",
                output_spread_scale=1.4,
            )
        )
    return RCMLQSelector(
        RCMLQConfig(
            resolution_min=0.2,
            resolution_max=6.0,
            resolution_steps=task.profile.rc_steps,
            posterior_samples=task.profile.rc_samples,
            beta=beta,
            coordinate_clip=None,
        )
    )


def _choice_probabilities(
    target: np.ndarray, queries: np.ndarray, beta_user: float
) -> np.ndarray:
    logits = -float(beta_user) * np.sum(
        (queries - target[None, :]) ** 2, axis=1
    )
    logits -= float(np.max(logits))
    probabilities = np.exp(logits)
    return probabilities / float(probabilities.sum())


def _synthetic_ratings(
    target: np.ndarray, queries: np.ndarray, probabilities: np.ndarray, winner: int
) -> tuple[int, int]:
    selected_distance = float(np.linalg.norm(queries[winner] - target))
    closeness = math.exp(-(selected_distance**2) / (2.0 * len(target)))
    preference = int(np.clip(round(1.0 + 9.0 * closeness), 1, 10))
    entropy = -float(
        np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, None)))
    )
    normalized_entropy = entropy / max(math.log(len(probabilities)), 1e-12)
    difficulty = int(np.clip(round(1.0 + 6.0 * normalized_entropy), 1, 7))
    return preference, difficulty


def _adapt_beta(beta: float, preference: int, difficulty: int) -> float:
    # Exact bounded log-space rule in EntropyQueryStrategy._adapt_beta.
    beta_min, beta_max, smoothing = 0.35, 2.5, 0.30
    closeness = np.clip((float(preference) - 1.0) / 9.0, 0.0, 1.0)
    ease = np.clip((7.0 - float(difficulty)) / 6.0, 0.0, 1.0)
    reliability = 0.35 * closeness + 0.65 * ease
    target = beta_min * (beta_max / beta_min) ** reliability
    updated = np.exp(
        (1.0 - smoothing) * np.log(beta) + smoothing * np.log(target)
    )
    return float(np.clip(updated, beta_min, beta_max))


def _pairwise_geometry(queries: np.ndarray) -> tuple[float, float]:
    distances = np.linalg.norm(
        queries[:, None, :] - queries[None, :, :], axis=-1
    )
    pairs = distances[np.triu_indices(len(queries), k=1)]
    return float(np.min(pairs)), float(np.max(pairs))


def _weighted_distance(
    particles: np.ndarray, weights: np.ndarray, point: np.ndarray
) -> float:
    return float(weights @ np.linalg.norm(particles - point[None, :], axis=1))


def run_task(task: Task) -> tuple[dict[str, object], list[dict[str, object]]]:
    target, persona = _scenario(task)
    beta_user = USER_BETA[task.noise]
    beta_model = beta_user * MODEL_MULTIPLIER[task.mismatch]
    posterior = ParticleMixturePreferencePosterior.from_gaussian_mixture(
        local_mean=persona,
        local_covariance=0.35 * np.eye(12),
        global_mean=np.zeros(12),
        global_covariance=np.eye(12),
        global_weight=0.45,
        particle_count=task.profile.particle_count,
        seed=stable_seed("particles", task.seed),
        beta=beta_model,
        metric=np.eye(12),
    )
    selector = _selector(task, beta_model)
    choice_random = np.random.default_rng(stable_seed("choice", task.seed))
    choice_uniforms = choice_random.random(task.profile.rounds)
    recovery_threshold = 1.5
    target_neighborhood_radius = 2.5
    initial_error = float(np.linalg.norm(posterior.mean - target))
    initial_target_mass = float(
        posterior.weights[
            np.linalg.norm(posterior.particles - target[None, :], axis=1)
            <= target_neighborhood_radius
        ].sum()
    )
    first_threshold_round: int | None = None
    would_stop_round: int | None = None
    low_ig_streak = 0
    any_resampled = False
    global_extinct_round: int | None = None
    rounds: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for round_index in range(1, task.profile.rounds + 1):
        acquisition = acquisition_posterior(
            posterior,
            np.zeros(12),
            np.eye(12),
            enabled=task.exploration_rho > 0.0,
            global_weight=task.exploration_rho,
        )
        select_started = time.perf_counter()
        query_seed = stable_seed("query", task.seed, round_index)
        if task.algorithm == "entropy":
            result = selector.select(
                acquisition,
                posterior.prior_covariance,
                task.m_value,
                seed=query_seed,
            )
        else:
            result = selector.select(acquisition, task.m_value, seed=query_seed)
        queries = np.asarray(result.query_points, dtype=np.float64)
        optimization_latency_ms = (time.perf_counter() - select_started) * 1000.0
        predicted_eig = expected_information_gain(
            posterior.particles, posterior.weights, queries, posterior.beta
        )
        probabilities = _choice_probabilities(target, queries, beta_user)
        winner = int(np.searchsorted(np.cumsum(probabilities), choice_uniforms[round_index - 1]))
        winner = min(winner, len(probabilities) - 1)
        preference, difficulty = _synthetic_ratings(
            target, queries, probabilities, winner
        )
        previous_beta = posterior.beta
        if task.adaptive_beta:
            posterior.beta = _adapt_beta(previous_beta, preference, difficulty)
        update_started = time.perf_counter()
        posterior.update(queries, winner)
        update_latency_ms = (time.perf_counter() - update_started) * 1000.0
        any_resampled = any_resampled or posterior.last_resampled
        if posterior.global_mass == 0.0 and global_extinct_round is None:
            global_extinct_round = round_index

        target_error = float(np.linalg.norm(posterior.mean - target))
        if target_error < recovery_threshold and first_threshold_round is None:
            first_threshold_round = round_index
        eigenvalues = np.linalg.eigvalsh(posterior.covariance)
        particle_entropy = -float(
            np.sum(posterior.weights * np.log(np.clip(posterior.weights, 1e-300, None)))
        )
        min_distance, max_distance = _pairwise_geometry(queries)
        query_radius = float(
            np.max(np.linalg.norm(queries - posterior.mean[None, :], axis=1))
        )
        target_mass = float(
            posterior.weights[
                np.linalg.norm(posterior.particles - target[None, :], axis=1)
                <= target_neighborhood_radius
            ].sum()
        )
        low_ig_streak = low_ig_streak + 1 if predicted_eig < 0.01 else 0
        if would_stop_round is None and (
            float(np.trace(posterior.covariance)) < 0.5 or low_ig_streak >= 2
        ):
            would_stop_round = round_index
        rounds.append(
            {
                "seed": task.seed,
                "persona_quality": task.persona_quality,
                "persona_target_distance": PERSONA_DISTANCE[task.persona_quality],
                "m_value": task.m_value,
                "algorithm": task.algorithm,
                "noise": task.noise,
                "mismatch": task.mismatch,
                "adaptive_beta": task.adaptive_beta,
                "exploration_rho": task.exploration_rho,
                "round": round_index,
                "target_error": target_error,
                "recovered": target_error < recovery_threshold,
                "posterior_particle_entropy": particle_entropy,
                "posterior_trace": float(np.trace(posterior.covariance)),
                "posterior_largest_eigenvalue": float(eigenvalues[-1]),
                "ess": posterior.effective_sample_size,
                "ess_before_resample": posterior.ess_before_resample,
                "resampled": posterior.last_resampled,
                "global_component_mass": posterior.global_mass,
                "target_neighborhood_mass": target_mass,
                "weighted_particle_distance_to_target": _weighted_distance(
                    posterior.particles, posterior.weights, target
                ),
                "weighted_particle_distance_to_persona": _weighted_distance(
                    posterior.particles, posterior.weights, persona
                ),
                "predicted_eig": predicted_eig,
                "cumulative_eig": float(
                    predicted_eig
                    + (rounds[-1]["cumulative_eig"] if rounds else 0.0)
                ),
                "query_radius": query_radius,
                "min_pairwise_theta_distance": min_distance,
                "max_pairwise_theta_distance": max_distance,
                "physical_resolution": float(
                    getattr(result, "physical_resolution", float("nan"))
                ),
                "beta_user": beta_user,
                "beta_model_before": previous_beta,
                "beta_model_after": posterior.beta,
                "preference_rating": preference,
                "difficulty_rating": difficulty,
                "optimization_latency_ms": optimization_latency_ms,
                "update_latency_ms": update_latency_ms,
                "render_latency_ms": float("nan"),
                "total_query_latency_ms": optimization_latency_ms + update_latency_ms,
                "would_stop": would_stop_round is not None,
                "would_stop_round": would_stop_round,
            }
        )
    final = rounds[-1]
    target_support_lost = bool(
        initial_target_mass >= 0.01
        and float(final["target_neighborhood_mass"]) < 0.1 * initial_target_mass
    )
    worse_than_initial = float(final["target_error"]) > initial_error
    run = {
        "seed": task.seed,
        "persona_quality": task.persona_quality,
        "persona_target_distance": PERSONA_DISTANCE[task.persona_quality],
        "m_value": task.m_value,
        "algorithm": task.algorithm,
        "noise": task.noise,
        "mismatch": task.mismatch,
        "adaptive_beta": task.adaptive_beta,
        "exploration_rho": task.exploration_rho,
        "beta_user": beta_user,
        "initial_beta_model": beta_model,
        "final_beta_model": posterior.beta,
        "initial_error": initial_error,
        "final_error": final["target_error"],
        "recovered": float(final["target_error"]) < recovery_threshold,
        "rounds_to_threshold": first_threshold_round,
        "censored": first_threshold_round is None,
        "catastrophic_failure": worse_than_initial or target_support_lost,
        "worse_than_initial": worse_than_initial,
        "target_support_lost": target_support_lost,
        "initial_target_neighborhood_mass": initial_target_mass,
        "final_target_neighborhood_mass": final["target_neighborhood_mass"],
        "initial_global_mass": 0.45,
        "final_global_mass": posterior.global_mass,
        "global_support_extinct": global_extinct_round is not None,
        "global_extinct_round": global_extinct_round,
        "any_resampled": any_resampled,
        "final_posterior_trace": final["posterior_trace"],
        "final_largest_eigenvalue": final["posterior_largest_eigenvalue"],
        "final_ess": final["ess"],
        "cumulative_eig": final["cumulative_eig"],
        "mean_optimization_latency_ms": float(
            np.mean([row["optimization_latency_ms"] for row in rounds])
        ),
        "p95_optimization_latency_ms": float(
            np.percentile([row["optimization_latency_ms"] for row in rounds], 95)
        ),
        "trajectory_latency_ms": (time.perf_counter() - run_started) * 1000.0,
        "would_stop": would_stop_round is not None,
        "would_stop_round": would_stop_round,
    }
    return run, rounds


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _groups(
    rows: Iterable[dict[str, object]], fields: tuple[str, ...]
) -> dict[tuple[object, ...], list[dict[str, object]]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[field] for field in fields), []).append(row)
    return grouped


def _numeric_summary(values: list[float], prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(max(len(array), 1))
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_std": std,
        f"{prefix}_ci95_low": float(np.mean(array) - half_width),
        f"{prefix}_ci95_high": float(np.mean(array) + half_width),
    }


def summarize(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    fields = (
        "persona_quality",
        "m_value",
        "algorithm",
        "noise",
        "mismatch",
        "adaptive_beta",
        "exploration_rho",
    )
    output: list[dict[str, object]] = []
    for key, group in sorted(_groups(runs, fields).items(), key=lambda item: str(item[0])):
        row = dict(zip(fields, key))
        row["n"] = len(group)
        row.update(_numeric_summary([float(x["final_error"]) for x in group], "final_error"))
        row.update(
            _numeric_summary(
                [float(x["cumulative_eig"]) for x in group], "cumulative_eig"
            )
        )
        row.update(
            _numeric_summary(
                [float(x["mean_optimization_latency_ms"]) for x in group],
                "optimization_latency_ms",
            )
        )
        for name in (
            "recovered",
            "catastrophic_failure",
            "worse_than_initial",
            "target_support_lost",
            "global_support_extinct",
            "any_resampled",
            "would_stop",
        ):
            values = np.asarray([bool(x[name]) for x in group], dtype=np.float64)
            rate = float(np.mean(values))
            half = 1.96 * math.sqrt(rate * (1.0 - rate) / len(values))
            row[f"{name}_rate"] = rate
            row[f"{name}_ci95_low"] = max(0.0, rate - half)
            row[f"{name}_ci95_high"] = min(1.0, rate + half)
        reached = [float(x["rounds_to_threshold"]) for x in group if not x["censored"]]
        row["rounds_to_threshold_mean_uncensored"] = (
            float(np.mean(reached)) if reached else float("nan")
        )
        output.append(row)
    return output


def _mean_by_round(
    rows: list[dict[str, object]], field: str, algorithm: str, persona: str
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["algorithm"] == algorithm
        and row["persona_quality"] == persona
        and row["mismatch"] == "match"
    ]
    grouped = _groups(selected, ("round",))
    ordered = sorted(grouped.items(), key=lambda item: int(item[0][0]))
    x = np.asarray([int(group_key[0]) for group_key, _ in ordered], dtype=int)
    y = np.asarray(
        [np.mean([float(row[field]) for row in values]) for _, values in ordered]
    )
    return x, y


def save_figures(output: Path, runs: list[dict[str, object]], rounds: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"entropy": "#276FBF", "rc_mlq": "#D1495B"}

    def indicator(value: object) -> float:
        if isinstance(value, str):
            return float(value.strip().lower() == "true")
        return float(bool(value))
    for filename, field, ylabel in (
        ("error_vs_round.pdf", "target_error", "Target error (L2)"),
        ("ig_vs_round.pdf", "predicted_eig", "Predicted EIG (nats)"),
        (
            "diversity_vs_round.pdf",
            "min_pairwise_theta_distance",
            "Minimum pairwise theta distance",
        ),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
        for axis, persona in zip(axes, PERSONA_DISTANCE):
            for algorithm in ("entropy", "rc_mlq"):
                x, y = _mean_by_round(rounds, field, algorithm, persona)
                axis.plot(x, y, label=algorithm, color=colors[algorithm])
            axis.set_title(persona)
            axis.set_xlabel("Round")
            axis.grid(alpha=0.2)
        axes[0].set_ylabel(ylabel)
        axes[-1].legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output / filename)
        plt.close(figure)

    core = [row for row in runs if row["mismatch"] == "match"] or runs
    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    x = np.arange(len(PERSONA_DISTANCE))
    width = 0.36
    for offset, algorithm in zip((-width / 2, width / 2), ("entropy", "rc_mlq")):
        values = []
        for persona in PERSONA_DISTANCE:
            matching = [
                indicator(row["recovered"])
                for row in core
                if row["algorithm"] == algorithm
                and row["persona_quality"] == persona
            ]
            values.append(float(np.mean(matching)) if matching else float("nan"))
        axis.bar(x + offset, values, width, label=algorithm, color=colors[algorithm])
    axis.set_xticks(x, PERSONA_DISTANCE.keys())
    axis.set_ylim(0, 1)
    axis.set_ylabel("Recovery rate (final error < 1.5)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "recovery_rate.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for algorithm in ("entropy", "rc_mlq"):
        values = [
            float(row["mean_optimization_latency_ms"])
            for row in core
            if row["algorithm"] == algorithm
        ]
        axis.boxplot(
            values,
            positions=[0 if algorithm == "entropy" else 1],
            widths=0.5,
            tick_labels=[algorithm],
        )
    axis.set_yscale("log")
    axis.set_ylabel("Optimization latency / query (ms, log scale)")
    figure.tight_layout()
    figure.savefig(output / "runtime.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.0))
    for algorithm in ("entropy", "rc_mlq"):
        means = []
        for persona in PERSONA_DISTANCE:
            matching = [
                float(row["final_error"])
                for row in core
                if row["algorithm"] == algorithm
                and row["persona_quality"] == persona
            ]
            means.append(float(np.mean(matching)) if matching else float("nan"))
        axis.plot(
            list(PERSONA_DISTANCE.values()),
            means,
            marker="o",
            label=algorithm,
            color=colors[algorithm],
        )
    axis.axhline(1.5, color="black", linestyle="--", linewidth=1, label="recovery threshold")
    axis.set_xlabel("Persona-target standardized distance")
    axis.set_ylabel("Mean final target error")
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "persona_distance_stress.pdf")
    plt.close(figure)


def build_tasks(args: argparse.Namespace, profile: ComputeProfile) -> list[Task]:
    seeds = range(args.seed_start, args.seed_start + (args.seeds or profile.seeds))
    mismatches = tuple(args.mismatch)
    adaptive_values = (True, False) if args.beta_ablation else (args.adaptive_beta,)
    return [
        Task(
            seed,
            persona,
            m_value,
            algorithm,
            noise,
            mismatch,
            adaptive,
            float(rho),
            profile,
        )
        for persona in args.persona
        for m_value in args.m
        for algorithm in args.algorithm
        for noise in args.noise
        for mismatch in mismatches
        for adaptive in adaptive_values
        for rho in args.rho
        for seed in seeds
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired theta-space IDEAL-Finder algorithm audit.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/algorithm_audit")
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--persona", nargs="+", choices=PERSONA_DISTANCE, default=list(PERSONA_DISTANCE))
    parser.add_argument("--m", nargs="+", type=int, choices=(2, 4, 8), default=[2, 4, 8])
    parser.add_argument("--algorithm", nargs="+", choices=("entropy", "rc_mlq"), default=["entropy", "rc_mlq"])
    parser.add_argument("--noise", nargs="+", choices=USER_BETA, default=list(USER_BETA))
    parser.add_argument("--mismatch", nargs="+", choices=MODEL_MULTIPLIER, default=["match"])
    parser.add_argument("--adaptive-beta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta-ablation", action="store_true")
    parser.add_argument(
        "--rho",
        nargs="+",
        type=float,
        default=[0.0],
        help="Query-time global acquisition mixture weight(s).",
    )
    args = parser.parse_args()
    if args.seeds is not None and args.seeds < 1:
        parser.error("--seeds must be positive")
    if any(not 0.0 <= rho < 1.0 for rho in args.rho):
        parser.error("every --rho value must be in [0, 1)")
    args.output.mkdir(parents=True, exist_ok=True)
    profile = PROFILES[args.profile]
    tasks = build_tasks(args, profile)
    manifest = {
        "created_unix": time.time(),
        "baseline": baseline_fingerprint(),
        "profile_name": args.profile,
        "compute_profile": asdict(profile),
        "task_count": len(tasks),
        "paired_seed_design": [
            "theta_star",
            "persona",
            "choice_uniforms",
            "initial_particle_seed",
        ],
        "persona_distance": PERSONA_DISTANCE,
        "user_beta": USER_BETA,
        "model_multiplier": MODEL_MULTIPLIER,
        "recovery_threshold": 1.5,
        "target_neighborhood_radius": 2.5,
        "local_covariance_scale": 0.35,
        "global_mixture_weight": 0.45,
        "query_exploration_rho": args.rho,
        "rendering": "not_run_theta_space_only",
        "notes": [
            "map_estimate is the weighted particle mean in the current implementation",
            "audit/quick profiles preserve acquisition equations but cap optimizer compute",
            "synthetic ratings are deterministic functions of choice entropy and selected-query distance",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    started = time.perf_counter()
    runs: list[dict[str, object]] = []
    rounds: list[dict[str, object]] = []
    print(f"Starting {len(tasks)} trajectories with {args.workers} workers", flush=True)
    if args.workers == 1:
        iterator = (run_task(task) for task in tasks)
        for index, (run, round_rows) in enumerate(iterator, start=1):
            runs.append(run)
            rounds.extend(round_rows)
            if index % 25 == 0 or index == len(tasks):
                print(f"completed {index}/{len(tasks)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(run_task, task): task for task in tasks}
            for index, future in enumerate(as_completed(future_map), start=1):
                run, round_rows = future.result()
                runs.append(run)
                rounds.extend(round_rows)
                if index % 100 == 0 or index == len(tasks):
                    print(f"completed {index}/{len(tasks)}", flush=True)
    sort_key = lambda row: (
        str(row["persona_quality"]),
        int(row["m_value"]),
        str(row["algorithm"]),
        str(row["noise"]),
        str(row["mismatch"]),
        bool(row["adaptive_beta"]),
        float(row["exploration_rho"]),
        int(row["seed"]),
        int(row.get("round", 0)),
    )
    runs.sort(key=sort_key)
    rounds.sort(key=sort_key)
    summary = summarize(runs)
    _write_csv(args.output / "stress_test_runs.csv", runs)
    _write_csv(args.output / "stress_test_rounds.csv", rounds)
    _write_csv(args.output / "baseline_summary.csv", summary)
    _write_csv(
        args.output / "persona_recovery.csv",
        [
            row
            for row in summary
            if row["persona_quality"] in {"moderate", "strong"}
        ],
    )
    _write_csv(
        args.output / "runtime_summary.csv",
        [
            {
                key: row[key]
                for key in row
                if key in {
                    "persona_quality",
                    "m_value",
                    "algorithm",
                    "noise",
                    "mismatch",
                    "adaptive_beta",
                    "exploration_rho",
                    "n",
                }
                or key.startswith("optimization_latency_ms")
            }
            for row in summary
        ],
    )
    if args.beta_ablation:
        _write_csv(args.output / "beta_ablation.csv", summary)
    if len(args.rho) > 1 or any(rho > 0.0 for rho in args.rho):
        _write_csv(args.output / "exploration_ablation.csv", summary)
    save_figures(args.output, runs, rounds)
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["completed_runs"] = len(runs)
    manifest["completed_rounds"] = len(rounds)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"Saved {len(runs)} runs and {len(rounds)} round rows to {args.output} "
        f"in {manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
