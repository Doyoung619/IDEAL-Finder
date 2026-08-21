# IDEAL-Finder Algorithm Audit

> Post-audit production override (2026-08-21): following real-user inspection
> of late-round StyleGAN outputs, the live study now uses 8 rounds per M,
> acquisition-only global exploration with `rho=0.15`, a latent-distance floor,
> 50% cross-round novelty slots, and a reduced query-optimization budget. These
> UX/runtime changes intentionally
> differ from the frozen baseline evaluated below; the original manifest remains
> the historical audit reference.

Audit date: 2026-08-21 (Asia/Seoul)

## Current implementation

The frozen baseline is not identified by the Git commit alone because the audit
started from a dirty worktree. The reproducible identifier is:

```text
commit: 997a437dbdab80211ea7eb12e908ecadf151c31d
baseline_algorithm_version: rc_mlq_v4_randomized_entropy_spread14
algorithm_manifest_sha256: 12399223b39c4fc1d78b37412ec3ea361cca570044e6c0f9f1cc2f9885f3d14d
```

The compact baseline fingerprint and every research-file SHA-256 are committed
as `docs/algorithm_baseline_manifest.json`; the ignored full audit run also has
the same fingerprint in `outputs/algorithm_audit/manifest.json`. All core
research files except the recommended user-study config remain byte-for-byte
identical to that snapshot.

| Feature | Status | Code-based finding |
|---|---|---|
| Persona warm start | implemented | A selected persona theta initializes each block; required mode rejects missing initialization. |
| Persona/global mixture prior | implemented | Exact `0.55 N(theta_persona, 0.35 Sigma) + 0.45 N(theta_global, Sigma)` particle initialization. |
| Particle posterior | implemented | Weighted particles, M-way softmax likelihood update, ESS and component labels. |
| Entropy Query | implemented | Torch optimizer directly maximizes Monte Carlo mutual information. The slim NumPy fallback is only partial: it uses a Gaussian approximation and fixed `beta=0.5`. |
| RC-MLQ | implemented | Principal covariance direction, posterior projected quantiles, and an exact-EIG physical-resolution grid. |
| Adaptive beta | implemented | Optional bounded log-space rule using preference and difficulty ratings. |
| Liu-West resampling/rejuvenation | implemented | Deterministic regularized resampling below `ESS < 0.12 N`; labels follow sampled ancestors. |
| Query diversity guard | partially implemented | Perceptual guard and bounded corrections exist, but the study threshold is `null`; `latent_min_distance` is logged/configured but is not used as an intervention condition. |
| East-Asian conditional prior | partially implemented operationally | Config, builder, validation, and loader paths exist. This checkout has no prior, StyleGAN, FairFace, or real persona-pool artifacts, so the real path could not be executed. |
| Query-time global exploration | not implemented | It was absent at baseline. The offline candidate failed development criteria and was removed from production code/config. |
| Predictive-surprise / soft reopen | not implemented | No update, state, or acquisition execution path exists. |
| `final_query_points` consistency | implemented | The post-guard theta actually mapped/rendered for display is saved as `final_query_points.npy`. |
| Fixed-beta mode | implemented | `query.adaptive_beta_enabled: false` leaves beta fixed. |
| Diversity disable flag | implemented | `query_diversity.enabled: false` preserves query theta and keeps diagnostics only. |

`ParticleMixturePreferencePosterior.map_estimate` is currently the weighted
particle mean, not a particle MAP. Target-error metrics below therefore use the
weighted mean and retain the existing public name only for compatibility.

## Stress-test design

The audit uses the production posterior, likelihood update, Entropy acquisition
equations, RC-MLQ selector, and adaptive-beta rule. It runs in whitened 12-D theta
space without rendering.

- `theta_star ~ N(0, I_12)`.
- Persona direction is reproducibly orthogonalized against `theta_star`.
- Persona-target standardized distances are 0.5 (aligned), 2.5 (moderate), and
  5.0 (strong).
- Initial local covariance is `0.35 I`; global covariance is `I`; mixture mass
  remains 0.55/0.45.
- User choice is exactly `softmax(-beta_user ||q_m - theta_star||^2)`.
- User beta is 2.5 (low noise), 1.0 (medium), or 0.35 (high).
- Model beta is separately tested at 0.5x, 1x, and 2x user beta.
- `M` is 2, 4, or 8; algorithms are Entropy and RC-MLQ; every block has 12 rounds.
- The core matched-beta matrix has 54 conditions and 100 independent paired
  seeds per condition: 5,400 trajectories and 64,800 round rows.
- Fixed-beta adds 5,400 paired trajectories. Model mismatch adds 10,800.
- Exploration development uses seeds 0-49 only: 7,200 trajectories for four rho
  values. Since no rho passed development criteria, seeds 50-99 remained
  untouched and no held-out claim is made.
- A higher-compute confirmatory slice uses 1,024 particles, 256 Entropy MC
  samples, 3 restarts x 60 steps, and RC 2,048 samples x 120 resolutions for 20
  seeds per selected condition.

The audit CPU profile caps production Entropy compute at 64 MC samples, one
restart, and eight optimizer steps so that the full matrix is tractable. This
changes optimizer budget, not its objective. The higher-compute slice checks the
key aligned/strong conclusion. It is still not a production GPU benchmark.

Recovery means final weighted-mean error `< 1.5`. Target-neighborhood mass is
particle mass within radius 2.5. A catastrophic run is either final error worse
than initial error or loss of at least 90% of a nontrivial initial target mass.
The CSVs contain round-wise particle entropy, covariance trace/largest
eigenvalue, ESS, EIG, geometry, global mass, target mass, beta, and latency.

Synthetic adaptive-beta ratings are deterministic diagnostics: selected-query
distance maps to preference 1-10, and true response entropy maps to difficulty
1-7. Consequently, the beta result is evidence for this declared rating model,
not a substitute for a pilot with human rating distributions.

## Results

### Baseline convergence and recovery

The following rows aggregate 900 runs each (three M values x three noise levels x
100 seeds). Confidence intervals are normal 95% intervals over runs; the
condition-level CSV also reports mean, median, standard deviation, and 95% CI for
every 100-seed cell.

| Persona | Algorithm | Mean final error (95% CI) | Median | Recovery | Catastrophic |
|---|---:|---:|---:|---:|---:|
| aligned | Entropy | 1.012 (0.994-1.030) | 0.979 | 95.0% | 8.1% |
| aligned | RC-MLQ | 1.158 (1.135-1.181) | 1.107 | 86.9% | 17.3% |
| moderate | Entropy | 2.175 (2.144-2.206) | 2.121 | 4.1% | 52.9% |
| moderate | RC-MLQ | 2.172 (2.145-2.200) | 2.141 | 3.1% | 53.6% |
| strong | Entropy | 3.172 (3.123-3.221) | 3.166 | 0.0% | 49.7% |
| strong | RC-MLQ | 3.014 (2.969-3.060) | 3.001 | 0.1% | 38.8% |

Aligned recovery is good but not uniform: under high user noise, recovery falls
to 89.3% for Entropy and 69.0% for RC-MLQ. Strong-wrong recovery is effectively
absent. The higher-compute M=4/medium-noise slice confirms this: aligned recovery
is 20/20 Entropy and 14/20 RC, while strong recovery is 0/20 for both.

### Global support and resampling

Global component labels can become extinct because resampling copies ancestor
labels without reserving global particles. Extinction rates are:

| Persona | Entropy | RC-MLQ |
|---|---:|---:|
| aligned | 79.1% | 70.1% |
| moderate | 52.1% | 53.8% |
| strong | 1.8% | 2.6% |

This does not explain the strong-wrong failure. In that condition, observed
choices quickly favor the global particles: mean global label mass at round 12
is 0.980 for Entropy and 0.969 for RC-MLQ. The failure is localization within a
broad 12-D global component under a 12-round budget, not disappearance of that
component. In aligned blocks, global extinction is usually the correct selection
of the persona-local mode.

### Query geometry and late-stage collapse

Theta spacing contracts with posterior concentration. For M=8, mean minimum
pairwise theta distance changes from 0.880 to 0.601 for Entropy (-31.7%) and from
0.493 to 0.263 for RC-MLQ (-46.5%) between rounds 1 and 12. This is a geometric
warning, not proof of same-face perceptual collapse.

The checked-in mock diagnostic produced zero guard triggers in 8 enabled cases,
because the configured image-similarity threshold is `null`. Mock embeddings are
not valid evidence for real facial distinguishability. Real StyleGAN/OpenCLIP
validation was not run because all required model/prior artifacts are absent.

### Runtime

Runtime depends strongly on the deliberately capped audit profile. In the more
representative confirmatory CPU slice, Entropy optimization averages about 115
ms/query and RC-MLQ about 32 ms/query, approximately 3.6x apart. Rendering latency
and total GPU query latency are unavailable without the StyleGAN artifacts. These
numbers must not be presented as production CUDA latency.

### Early-stopping diagnostic

Using the declared diagnostic rule (`trace(cov) < 0.5` or EIG `< 0.01` for two
consecutive rounds), 50.9% of Entropy runs would stop at mean round 7.88 and 36.0%
of RC runs at mean round 10.06. The actual 12-round study behavior is unchanged.

## Baseline questions Q1-Q7

1. **Aligned convergence:** Entropy generally converges; RC-MLQ mostly converges
   but is notably weaker under high noise.
2. **Strong-wrong recovery:** No. Recovery is 0.0% Entropy and 0.1% RC-MLQ.
3. **Global support loss:** It can occur after resampling, but it is rare in the
   strong-wrong failure and therefore is not its cause.
4. **Late query collapse:** Theta spacing contracts, especially RC M=8. Real
   perceptual collapse remains unmeasured.
5. **Adaptive beta:** It has no robust benefit under the declared rating model
   and materially hurts several cells; fixed beta is cleaner and safer.
6. **Diversity guard information loss:** Current study config never intervenes,
   so observed IG loss is zero but efficacy is also untested. Accepted RC radius
   corrections are already bounded to at most 5% IG loss by code and tests.
7. **Worse failure mode:** RC is worse when aligned/high-noise; Entropy has higher
   strong-wrong error and catastrophic rate. Neither solves strong recovery.

## Ablations

### Candidate A: query-time global acquisition mixture

The candidate used the same acquisition mixture for both algorithms and never
changed posterior updates. Development criteria were fixed before reading the
results: aligned degradation <=5%; either +10 percentage points strong recovery
or a two-algorithm paired-error CI below zero; <=20% runtime increase; one rho for
all M.

| rho | Entropy aligned error | Entropy strong error | RC aligned error | RC strong error |
|---:|---:|---:|---:|---:|
| .025 | +2.0% | +0.5% | -8.1% | -0.4% |
| .05 | +2.5% | +0.9% | -28.6% | -0.6% |
| .10 | +0.7% | +1.0% | -40.2% | +0.6% |
| .20 | -4.9% | +2.4% | -46.4% | +1.2% |

No strong-wrong paired improvement had a 95% CI below zero for both algorithms,
and recovery did not improve by 10 points. No rho was selected, held-out seeds
were not opened, and the production candidate was removed. **Recommendation:
REJECT.**

### Candidate B: fixed versus adaptive beta

Values are fixed-minus-adaptive paired error. Negative favors fixed.

| Algorithm | Persona | Error delta (95% CI) | Recovery change | Catastrophic change |
|---|---|---:|---:|---:|
| Entropy | aligned | -0.001 (-0.026, +0.023) | +1.3 pp | +0.3 pp |
| Entropy | moderate | -0.045 (-0.080, -0.010) | -0.6 pp | -5.1 pp |
| Entropy | strong | -0.079 (-0.124, -0.034) | +0.2 pp | -6.1 pp |
| RC-MLQ | aligned | -0.172 (-0.199, -0.144) | +10.1 pp | -10.9 pp |
| RC-MLQ | moderate | -0.048 (-0.076, -0.019) | +0.1 pp | -3.8 pp |
| RC-MLQ | strong | -0.051 (-0.088, -0.014) | -0.1 pp | -4.7 pp |

Model-beta under/match/over aggregate final error is 2.087/2.120/2.153 for
Entropy and 2.087/2.115/2.135 for RC-MLQ. The adaptive rule pulls final beta
toward roughly 1.1, but does not improve robustness. It also creates an
algorithm x user-rating feedback loop. **Recommendation: RECOMMEND FIXED FOR
CLEANER EXPERIMENT.** The generic default remains unchanged for compatibility;
`configs/persona_study.yaml` now sets adaptive beta off and uses search version
`rc_mlq_v4_randomized_entropy_spread14_fixed_beta_audit20260821`.

### Candidate C: diversity guard

RC-MLQ already tries same-principal-direction radius factors 1.1, 1.2, and 1.35
before any orthogonal fallback, and accepts only when relative EIG loss is at most
0.05. No new correction was needed. Without real embeddings, selecting a
similarity threshold or claiming perceptual benefit would be unjustified.
**Recommendation: KEEP OPTIONAL / diagnostic-only until real validation.**

### Candidate D: early stopping

Only `would_stop` and `would_stop_round` are recorded in offline CSVs. The study
still executes 12 rounds. **Recommendation: DIAGNOSTIC ONLY.**

## Recommended final configuration

```yaml
query:
  beta: 1.0
  adaptive_beta_enabled: false

query_diversity:
  enabled: true
  image_similarity_threshold: null  # diagnostic-only until real calibration
  max_relative_ig_loss: 0.05
  radius_expansion_factors: [1.1, 1.2, 1.35]

persona:
  prior_covariance_scale: 0.35
  global_mixture_weight: 0.45
  global_covariance_scale: 1.0
```

No query-exploration flag is present because the candidate was rejected. The
posterior, mixture, Entropy, RC-MLQ, Liu-West, and 12-round behavior are unchanged.

| Modification | Benefit | Risk | Recommendation |
|---|---|---|---|
| Query-time global rho | No strong-wrong recovery benefit | Adds asymmetric algorithm effects | REJECT |
| Fixed beta | Lower error/catastrophic rate, especially RC | Synthetic rating-model dependence | ACCEPT for user-study config |
| Current diversity guard | Preserves RC 1-D geometry and caps IG loss | Real threshold unvalidated, currently inactive | KEEP OPTIONAL |
| Early stopping | Useful future-efficiency diagnostic | Would alter study exposure if enabled | KEEP DIAGNOSTIC ONLY |

## Rejected changes

- No epsilon-greedy query, covariance inflation, soft reopen, neural policy, 2-D
  RC-MLQ, M-specific tuning, or persona-mixture tuning was added.
- The query-time global safeguard was removed after failing development criteria.
- No diversity threshold was inferred from mock embeddings.
- No automatic combination of individually tested candidates was enabled.

## Regression evidence

- With production candidate code removed, the frozen core research file hashes
  match the baseline manifest exactly.
- Current slim environment: 69 passed, 3 skipped (Torch-only tests skipped).
- Current Torch environment algorithm/demographic subsets: 22 passed in two
  processes; combined collection exposed a local macOS NumPy/Torch native-runtime
  import conflict before project tests executed.
- The recommended persona-study config is asserted deterministic in
  `tests/test_persona_config.py`.

## Remaining limitations

- The real East-Asian PCA geometry and coordinate clip were unavailable; the
  synthetic audit uses its intended whitened `N(0, I)` representation.
- Entropy full production optimizer settings and GPU latency were not executed.
- No real StyleGAN/OpenCLIP contact sheets, FairFace validity metrics, or render
  latency can be produced from this checkout.
- Strong-wrong recovery remains an observed structural limitation. The tested
  tiny global acquisition mixture does not fix it, and adding a more complex
  mechanism immediately before the study has a worse benefit/risk ratio than
  documenting the limitation.

## Final decisions

1. **Does wrong Persona cause a real recovery problem?** Yes, severe under this
   12-D/12-round stress definition.
2. **Is query-time global exploration needed?** The tested fixed rho safeguard is
   not supported; no rho is recommended.
3. **Does adaptive beta help?** No under the declared paired simulation; fixed is
   recommended for the user study.
4. **How serious is late same-face behavior?** Theta contraction is measurable;
   real same-face severity is unknown without StyleGAN/OpenCLIP.
5. **Does RC correction unnecessarily break 1-D geometry?** No. The existing
   code expands on the principal axis before bounded orthogonal fallback.
6. **What is the compute difference?** About 3.6x Entropy/RC optimization latency
   in the higher-compute CPU slice; production GPU/render cost remains unknown.
7. **Must the algorithm change before the study?** There is evidence for one
   conservative config change, not a new acquisition algorithm.
8. **Best single benefit/risk change:** turn adaptive beta off in the user-study
   config and keep all posterior/query mathematics unchanged.
