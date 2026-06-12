from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from dataset_batch import discover_npz_files, iter_graphs, json_dumps, load_batch_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3 (light): compute fast sparse-layout invariants.")
    p.add_argument("--dataset-root", default="datasets", help="Root folder that contains N6/N7/N8/N9")
    p.add_argument("--out-dir", default="artifacts/niliso", help="Output directory")
    return p.parse_args()


def _sha1(payload: dict) -> str:
    return hashlib.sha1(json_dumps(payload).encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage3_light_invariants.csv"
    summary_path = out_dir / "stage3_light_summary.json"

    files = discover_npz_files(dataset_root)
    total = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "family",
                "source_file",
                "graph_index",
                "n",
                "nv",
                "nz",
                "entry_count",
                "nnz_i_unique",
                "nnz_j_unique",
                "nnz_k_unique",
                "k_support_in_v",
                "k_support_in_z",
                "k_support_ratio_z",
                "coef_abs_sum",
                "coef_abs_max",
                "coef_l1_histogram",
                "i_j_pair_count",
                "triplet_count_unique",
                "quick_hash",
            ],
        )
        writer.writeheader()

        for file_idx, npz_path in enumerate(files, start=1):
            print(f"[stage3-light] file {file_idx}/{len(files)}: {npz_path}", flush=True)
            family = npz_path.parent.name
            batch = load_batch_file(npz_path, compute_hash=False)
            for g in iter_graphs(batch):
                total += 1
                if total % 50000 == 0:
                    print(f"[stage3-light] processed graphs: {total}", flush=True)

                ii = g.i.astype(np.int64, copy=False)
                jj = g.j.astype(np.int64, copy=False)
                kk = g.k.astype(np.int64, copy=False)
                cc = g.c.astype(np.int64, copy=False)

                entry_count = int(ii.size)
                i_unique = int(np.unique(ii).size)
                j_unique = int(np.unique(jj).size)
                k_unique = int(np.unique(kk).size)

                k_support_in_v = int(np.sum(kk < g.nv))
                k_support_in_z = int(np.sum(kk >= g.nv))
                k_support_ratio_z = float(k_support_in_z / entry_count) if entry_count > 0 else 0.0

                abs_cc = np.abs(cc)
                coef_abs_sum = int(abs_cc.sum()) if abs_cc.size else 0
                coef_abs_max = int(abs_cc.max()) if abs_cc.size else 0

                # compact coefficient histogram for 
                # quick bucket split without heavy algebraic computation.
                coef_hist = {}
                if abs_cc.size:
                    vals, cnts = np.unique(abs_cc, return_counts=True)
                    coef_hist = {str(int(v)): int(c) for v, c in zip(vals, cnts)}

                pair_count = int(np.unique(np.stack([ii, jj], axis=1), axis=0).shape[0]) if entry_count else 0
                triplet_count = (
                    int(np.unique(np.stack([ii, jj, kk], axis=1), axis=0).shape[0])
                    if entry_count
                    else 0
                )

                payload = {
                    "n": int(g.n),
                    "nv": int(g.nv),
                    "nz": int(g.nz),
                    "entry_count": entry_count,
                    "k_unique": k_unique,
                    "k_support_ratio_z": round(k_support_ratio_z, 8),
                    "coef_abs_sum": coef_abs_sum,
                    "coef_abs_max": coef_abs_max,
                    "pair_count": pair_count,
                    "triplet_count": triplet_count,
                    "coef_hist": coef_hist,
                }

                writer.writerow(
                    {
                        "graph_id": g.graph_id,
                        "family": family,
                        "source_file": npz_path.as_posix(),
                        "graph_index": int(g.graph_index),
                        "n": int(g.n),
                        "nv": int(g.nv),
                        "nz": int(g.nz),
                        "entry_count": entry_count,
                        "nnz_i_unique": i_unique,
                        "nnz_j_unique": j_unique,
                        "nnz_k_unique": k_unique,
                        "k_support_in_v": k_support_in_v,
                        "k_support_in_z": k_support_in_z,
                        "k_support_ratio_z": f"{k_support_ratio_z:.8f}",
                        "coef_abs_sum": coef_abs_sum,
                        "coef_abs_max": coef_abs_max,
                        "coef_l1_histogram": json.dumps(coef_hist, ensure_ascii=False, sort_keys=True),
                        "i_j_pair_count": pair_count,
                        "triplet_count_unique": triplet_count,
                        "quick_hash": _sha1(payload),
                    }
                )

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "npz_file_count": len(files),
        "graphs_processed": total,
        "invariants_csv": csv_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
