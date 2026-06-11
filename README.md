# sol_Lie

Solvable Lie algebra isomorphism pipeline workspace, with optimized filtering/certification flow and handoff materials for the next phase focused on 2-step nilpotent Lie algebras.

## Current Status

This repository currently contains:

1. Main pipeline implementation in `isomorphism_pipeline.py`.
2. Baseline snapshots for reproducibility:
   1. `isomorphism_pipeline_base_before_5steps.py`
   2. `isomorphism_pipeline_baseline_after_5steps_v2.py`
3. Batch and pair reports from optimization/validation runs.
4. Session handoff documents for continuity.

Current state summary:

1. Staged batch flow (light bucket -> collision -> optional deepening) is in place.
2. Exact sparse monomial search is integrated as certification path.
3. ROI and next-run suggestion/template generation are supported.
4. Dedup-oriented refactors in analyze/profile paths are applied.

## Quick Start

### 1) Environment

Requirements:

1. Python 3.10+ (tested on conda Python 3.13.x)
2. `sympy`

Install dependency:

```bash
pip install sympy
```

### 2) Pair Mode Example

```bash
python isomorphism_pipeline.py \
  --alg1 compute_derivations.py \
  --alg2 invariant_extractor_B.py \
  --n 23 \
  --out isomorphism_report_AB_manual2.json \
  --eq-preview 500 \
  --system-max-pairs 500 \
  --invariant-level quick \
  --extension-profile-level quick \
  --same-kernel-mode \
  --hybrid-solve \
  --modp-primes 101,103,107,109,113 \
  --numeric-max-vars 40 \
  --numeric-max-eqs 80 \
  --numeric-restarts 30
```

### 3) Batch Mode Example

```bash
python isomorphism_pipeline.py \
  --batch-glob "*extractor*.py" \
  --n 23 \
  --out batch_report_opt7_flow_v2.json \
  --bucket-deepen-min-size 3 \
  --pencil-basis-cap 3 \
  --pencil-basis-cap-deep 6
```

## Reproducible Benchmark Protocol

For statistically meaningful A/B checks:

1. Use the same input and CLI arguments.
2. Run interleaved baseline/optimized trials (e.g., 10 + 10).
3. Compare median and quartiles, not single-run time.

Existing benchmark artifacts:

1. `.bench_tmp/ab_runs_raw.csv`
2. `.bench_tmp/ab_summary.json`

## Key Files

Core code:

1. `isomorphism_pipeline.py`

Baselines and planning:

1. `isomorphism_pipeline_base_before_5steps.py`
2. `isomorphism_pipeline_baseline_after_5steps_v2.py`
3. `isomorphism_pipeline_baseline_after_5steps_v2_features.md`
4. `optimization_next6_plan.md`

Session and handoff docs:

1. `CHAT_SESSION_READABLE_20260611.md`
2. `NEXT_ROUND_2STEP_NILPOTENT_DETAILED_GUIDE.md`
3. `IMPORTANT_FILES_FOR_REMOTE_PUSH.md`

Representative outputs:

1. `isomorphism_report_AB_opt7_flow.json`
2. `isomorphism_report_AB_dedup_recompute_v2.json`
3. `batch_report_opt7_flow_v2.json`
4. `isomorphism_report_AB_manual2.json`

## Next Round: 2-Step Nilpotent Plan

Primary objective:

1. Transition from general solvable-extension optimization to a specialized isomorphism workflow for 2-step nilpotent Lie algebras.

Immediate tasks:

1. Add dedicated 2-step mode switch.
2. Build 2-step-specific light hash and collision analysis.
3. Add stricter orbit-aware deep signatures under 2-step mode.
4. Preserve exact-stage correctness and re-run A/B protocol per major change.

Detailed roadmap:

1. See `NEXT_ROUND_2STEP_NILPOTENT_DETAILED_GUIDE.md`.

## Notes

1. This repository is intended for iterative research engineering; report files are intentionally kept for traceability.
2. If runtime behavior changes after updates, regenerate representative reports and update benchmark artifacts.
