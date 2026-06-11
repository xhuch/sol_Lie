# Next Round Guide: 2-Step Nilpotent Lie Algebra Isomorphism

Date: 2026-06-11
Status: Prepared handoff document

## 1. Target Scope

Focus next iteration on isomorphism identification specialized for 2-step nilpotent Lie algebras, rather than general solvable extension pipeline behavior.

Core shape:

1. [n, n] subset Z(n)
2. Bracket map can be treated as alternating bilinear map into center
3. Classification signal often dominated by induced tensor/pencil signatures and orbit structure under admissible basis changes

## 2. Current Pipeline Assets You Can Reuse

Main file:

1. isomorphism_pipeline.py

Reusable building blocks already present:

1. adapted basis workflow and profile context caching
2. fast 2-step bracket tensor path
3. matrix-group signature, extension action profile, matrix-pencil signature
4. exact sparse monomial search with failure-memory pruning
5. batch bucketing/deepening/ROI framework
6. report and recommendation artifact generation

## 3. Proposed Technical Plan for 2-Step Specialization

### Phase A: Normal Form and Structural Invariants

1. Center and derived consistency checks as hard prefilters.
2. Canonicalized block decomposition around V + Z partition.
3. Stronger invariants from alternating form tuple over V -> Z:
   1. rank profile of component forms
   2. joint kernel/image dimensions
   3. commutator-derived compatibility signatures

### Phase B: Orbit-Aware Signature Refinement

1. Use robust signature families that survive admissible basis transforms.
2. Build hashable signature ladder from cheap to expensive:
   1. rank-only and support-only summaries
   2. pairwise and sampled linear combination signatures
   3. pencil-level signatures with adaptive basis cap

### Phase C: Exact Certification Path

1. Keep exact sparse monomial search as certifier.
2. Introduce 2-step-specific pruning rules:
   1. center-preserving constraints first
   2. V-block action compatibility before full expansion
3. Preserve failure-memory cache semantics.

### Phase D: Batch Throughput Strategy

1. Keep light bucket hash but bias for 2-step invariants.
2. Trigger deepening only on collision buckets beyond threshold.
3. Add metrics for false-collision rate under 2-step mode.

## 4. Correctness and Benchmark Protocol

### Correctness

1. Maintain current final decision contract.
2. Add focused regression set for 2-step families:
   1. isomorphic pair permutations
   2. near-miss non-isomorphic pairs with same basic dims

### Performance

Use stable protocol per change:

1. 10 interleaved baseline/optimized runs.
2. Compare median plus p25/p75.
3. Treat gains as valid only if distribution separation is clear.

## 5. Important Existing Files to Carry Forward

Code:

1. isomorphism_pipeline.py
2. isomorphism_pipeline_baseline_after_5steps_v2.py
3. isomorphism_pipeline_base_before_5steps.py

Reports and benchmarks:

1. isomorphism_report_AB_opt7_flow.json
2. isomorphism_report_AB_dedup_recompute_v2.json
3. batch_report_opt7_flow_v2.json
4. .bench_tmp/ab_summary.json
5. .bench_tmp/ab_runs_raw.csv

Planning/reference:

1. optimization_next6_plan.md
2. isomorphism_pipeline_baseline_after_5steps_v2_features.md
3. CHAT_SESSION_READABLE_20260611.md

## 6. Suggested First Tasks Next Session

1. Add a dedicated CLI switch for 2-step-specialized mode.
2. Implement a 2-step-specific light hash and compare collision behavior.
3. Add one stricter orbit-aware deep signature under that mode.
4. Run A/B protocol and update ROI template logic accordingly.

## 7. Risk Checklist

1. Avoid overfitting to current A/B sample pair only.
2. Keep fallback path when specialized assumptions fail.
3. Ensure report compatibility for existing tooling.
4. Keep semantic equivalence guarantees in exact stage.
