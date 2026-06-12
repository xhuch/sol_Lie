# Session Checkpoint (2026-06-12)

This checkpoint is created before repository migration from D to F to preserve planning context and execution state.

## 1) Strategic Roadmap (Locked)

1. Phase A: nilpotent Lie algebra isomorphism identification.
2. Phase B: generate all 2D solvable extensions from nilpotent representatives.
3. Phase C: solvable isomorphism deduplication using the existing pipeline core.

Critical prior fact used:

1. Algebras from different source NPZ files are guaranteed non-isomorphic.
2. Therefore candidate generation only needs intra-file comparisons.

## 2) Dataset Contract (Locked)

Batch sparse NPZ layout per file:

1. graph_ptr: segment offsets for each graph.
2. i, j, k, c: sparse structure-constant entries.
3. graph_nv, graph_nz, graph_dim: per-graph dimensions.
4. conventions:
   1. zero-based indexing
   2. only i < j stored
   3. basis order V_then_Z

Observed keys in sample NPZ also include metadata:

1. source_mat_file
2. source_variable
3. num_graphs
4. index_base
5. antisymmetric
6. basis_order
7. storage_convention

## 3) Completed Pipeline Step-1 Deliverables

Implemented scripts in niliso_pipeline:

1. dataset_batch.py
2. stage1_build_manifest.py
3. stage2_validate_dataset.py
4. stage3_compute_invariants.py (heavy symbolic variant, kept but not used for full run)
5. stage3_compute_light_invariants.py (full-scale first-pass invariant extraction)

Produced artifacts in artifacts/niliso:

1. stage1_manifest.csv
2. stage1_summary.json
3. stage2_validation.csv
4. stage2_validation_summary.json
5. stage3_light_invariants.csv
6. stage3_light_summary.json

## 4) Current Quantitative State

From stage summaries:

1. NPZ files: 103
2. Graphs: 2,382,296
3. Family file counts: N6=7, N7=14, N8=28, N9=54
4. Validation checks total: 9,530,111
5. Validation failures: 0
6. Light invariants processed: 2,382,296

## 5) Next Immediate Task (Phase A continuation)

1. Stage 4 candidate generation using stage3_light_invariants.csv
2. Intra-file only candidate pairing
3. candidate-size report per file and per bucket

## 6) Session Preservation Notes

1. Chat UI history in VS Code may not migrate with folder path changes.
2. Project-critical context is preserved in repository files:
   1. CHAT_SESSION_READABLE_20260611.md
   2. NEXT_ROUND_2STEP_NILPOTENT_DETAILED_GUIDE.md
   3. SESSION_CHECKPOINT_20260612.md (this file)
   4. README.md
3. After migration, open the new folder and continue from Stage 4.
