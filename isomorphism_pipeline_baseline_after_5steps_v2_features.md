# Baseline Features: isomorphism_pipeline_baseline_after_5steps_v2.py

Snapshot file:
- isomorphism_pipeline_baseline_after_5steps_v2.py

Captured time:
- 2026-06-11 (before the next 6-step optimization round)

Feature profile (this baseline includes the original 5-step optimization set):

1. Shared profile context
- build_adapted_profile_context is present and reused by profile stages.

2. Pair parallel option
- pair-parallel-workers exists (default 1).

3. mod-p profile prefilter path
- profile-modp-mode and profile-modp-primes exist.
- mod-p signatures for bracket/extension are available.

4. Cross-run artifact cache
- artifact-cache-mode and artifact-cache-path exist.
- report includes artifact_cache hit/miss counters.

5. Aut(n)-oriented signature default-on
- autn_oriented_signature is integrated in linear checks.
- hidden developer emergency switch exists: --dev-disable-autn-signature.

Known behavior at this baseline stage:
- Good rejection power for batch tasks.
- Pair latency may increase on isomorphic pairs due to stronger profile signatures.
- No post-5 tuning yet (before adaptive 6-step follow-up).
