# IDEAL-Finder user-study release

Release date: 2026-08-21 (Asia/Seoul)

This document is the canonical algorithm/configuration freeze for the Persona
user study. The study configuration is `configs/persona_study.yaml`; do not make
a second “final” config.

## Final study algorithm

```text
Persona questionnaire
→ 8 deterministic Persona candidates (at most one non-repeating reroll)
→ selected Persona + analysis-only alignment rating
→ 0.55 Persona-local + 0.45 global conditional particle prior
→ six counterbalanced M × algorithm blocks
→ 8 Bayesian selection rounds per block
→ final evaluation and survey
```

The selected Persona initializes every block. Entropy Query and RC-MLQ use the
same seeded initial particles/weights inside a matched condition. The particle
likelihood/update, Liu–West resampling/rejuvenation, and StyleGAN theta-to-W
mapping remain unchanged. The post-audit production override changes query
acquisition/novelty, runtime budget, and round count as recorded below.

## Final configuration

| Item | Frozen value |
|---|---|
| Canonical file | `configs/persona_study.yaml` |
| M values | `8, 4, 2` |
| Algorithms | `entropy`, `rc_mlq` |
| Schedule | 6 reproducibly counterbalanced blocks |
| Rounds | 8 per block |
| Persona/global mixture | `0.55 / 0.45` |
| Local/global covariance scale | `0.35 / 1.0` |
| Particles | 8,192 |
| Model beta | fixed `1.0` |
| Adaptive beta | OFF (`adaptive_beta_enabled: false`) |
| Query-time global exploration | acquisition-only `rho=0.15` |
| Cross-round novelty | 50% slots, deterministic global pool of 64 |
| Entropy budget | 256 posterior samples, 3 restarts × 60 steps; spread `1.4` |
| RC-MLQ budget | 2,048 samples × 80 resolution steps |
| Diversity | latent floor `0.75`; real-image threshold `null` |
| Persona candidates | 8 initially, at most one reroll |
| Persona condition | gender-specific, age 20–29, East Asian operational filter |

With a `null` OpenCLIP similarity threshold, no uncalibrated perceptual
similarity correction is automatically activated. RC correction order remains
same-principal-direction radius expansion followed by the existing orthogonal
fallback. The live acquisition view mixes 15% global conditional prior samples
without changing the stored Bayesian posterior. From round 2, deterministic
farthest-global samples replace up to 50% of slots that are closest to earlier
queries. The original offline candidate script remains historical audit tooling;
the production implementation is `core/exploration.py` and
`core/query_novelty.py`.

The Persona confirmation page already records the 1–10 response to “이 얼굴은
현재 이상형에 얼마나 가까운가요?” and a 1–7 selection-confidence response.
They are analysis metadata only and are included in the normalized CSV export;
fixed beta means they do not change the posterior update. Candidate batches log
page/order/pool IDs, selected pool/image ID, and per-image generator seed. The
page seed is deterministically derived from participant ID, base seed, and page.

## Audit evidence

The frozen baseline audit covered 5,400 trajectories, 64,800 rounds, and 100
paired seeds per condition. Strongly-wrong Persona recovery was 0.0% for
Entropy and 0.1% for RC-MLQ. Global component mass loss was not the cause;
remaining global mass localized too slowly in 12 dimensions under the fixed
12-round budget.

The original fixed-rho candidate failed its synthetic development criterion;
that historical conclusion remains valid for the 12-round audit. Fixed beta was
the only algorithm-level change supported by that paired simulation. For
aligned RC-MLQ it changed error by -0.172, recovery by +10.1 percentage points,
and catastrophic failure by -10.9 percentage points. The later production
override was made after real-user inspection of late-round StyleGAN output; it
is an operational UX/runtime decision and was not revalidated by the original
5,400-run matrix. Full historical evidence is in `docs/ALGORITHM_AUDIT.md`.

## Important deviation from baseline

Commit `52206939a56d7750949bfa295e90d86cdc9fea96` intentionally supersedes the
earlier fixed 12-round study config. Its deviations are fixed beta, 8 rounds,
acquisition-only global `rho=0.15`, cross-round novelty slots, latent-distance
floor `0.75`, and reduced Entropy/RC compute budgets. The Bayesian update and
stored posterior remain unchanged; no reopen controller or adaptive beta was
added.

## Release validation

- Python 3.10 slim suite: 69 passed, 3 skipped. The skips are exactly the three
  Torch-optional collection modules (`test_entropy_query`,
  `test_demographic_age`, `test_demographic_prior_builder`).
- Separate Torch environment algorithm/demographic subsets: 22 passed. They were
  run in two processes because one combined collection order triggered a local
  macOS NumPy/Torch native-runtime abort before project tests executed.
- Final gateway/mock-GPU flow, one Persona reroll, confirmation, session
  creation, rounds, persistence, refresh/resume, completion, CSV export,
  participant isolation, and GPU authentication: passed.
- Python compile, every Bash script syntax check, dependency consistency, and
  `git diff --check`: passed.
- Baseline fingerprint comparison excludes the documented post-audit production
  files listed in `tests/test_persona_config.py`; all other audited files match.

## Known limitations

- Strongly-wrong Persona initialization was not reliably recovered in the audited 12-round baseline; the 8-round override has no equivalent full-matrix recovery claim.
- The global prior component does not solve that localization problem by itself.
- Acquisition exploration/novelty were enabled from real-user UX inspection despite the synthetic rho ablation failing its original criterion.
- Real StyleGAN/OpenCLIP similarity calibration and GPU latency/VRAM/disk measurement require the target lab artifacts and server.
- The diversity threshold therefore remains `null`; mock embeddings are not evidence for perceptual calibration.
- Persona-only design cannot establish a causal advantage over a no-Persona control.
- FairFace/OpenCLIP labels and scores are operational filters with model bias, not participant identity judgments.

## Freeze policy

After deployment verification, the researcher records the release commit and
config hash and starts collection without further algorithm or config changes.
Any later change requires a new version, new hash, documented rationale, and a
separate study decision; do not mix resulting sessions into this release.

## Reproducibility

```text
Baseline algorithm commit:
997a437dbdab80211ea7eb12e908ecadf151c31d

Baseline algorithm manifest SHA-256:
12399223b39c4fc1d78b37412ec3ea361cca570044e6c0f9f1cc2f9885f3d14d

Final release commit:
the commit containing this document (resolve with: git rev-parse HEAD)

Post-audit production override source commit:
52206939a56d7750949bfa295e90d86cdc9fea96

Study config filename:
configs/persona_study.yaml

Study config SHA-256:
2a225eb9355af0f4198088e0340329e31bd0ec1fbbdb90f53631900634101b71
```

Use the exact release commit as `IDEAL_APP_VERSION`. Verify the checkout before
deployment:

```bash
git rev-parse HEAD
shasum -a 256 configs/persona_study.yaml
git status --short
```

The config hash must equal the value above and the status output must be empty.
GPU setup is in `docs/GPU_SERVER_HANDOFF.md`; researcher-owned production steps
are in `docs/DEPLOYMENT_CHECKLIST.md`.
