# Next 6 Optimization Steps (Post 5-step Version)

1. Adaptive mod-p prefilter enablement
- Goal: avoid mod-p overhead on likely-isomorphic collision pairs; keep it for likely rejection cases.
- Strategy: `profile_modp_mode=auto` with lightweight heuristics from base invariants.

2. Layered prime early-stop for mod-p
- Goal: stop as soon as mismatch appears.
- Strategy: compare signatures prime-by-prime and terminate immediately on first mismatch.

3. Matrix-pencil invariant on 2-step kernel
- Goal: stronger linear-stage rejection for 2-step nilpotent kernel.
- Strategy: sample rank-signatures of linear combinations of central 2-forms.

4. Exact sparse search branching order refinement
- Goal: shrink backtracking tree.
- Strategy: branch by (candidate count, higher local degree first).

5. Batch process-level parallelism
- Goal: improve throughput on multi-file batch mode.
- Strategy: optional ProcessPool worker path (cache-off by default path), sequential fallback.

6. Two-level cache + version stamp
- Goal: robust caching across runs and within run.
- Strategy: L1 in-memory map + L2 disk JSON cache with schema/version tag.
