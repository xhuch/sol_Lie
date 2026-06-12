from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import List

import sympy as sp

from dataset_batch import dense_structure_constants, discover_npz_files, iter_graphs, json_dumps, load_batch_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: compute fast invariants for nilpotent graphs.")
    p.add_argument("--dataset-root", default="datasets", help="Root folder that contains N6/N7/N8/N9")
    p.add_argument("--out-dir", default="artifacts/niliso", help="Output directory")
    p.add_argument("--max-graphs", type=int, default=0, help="Optional cap for quick smoke test (0 means all)")
    return p.parse_args()


def _span_rank(vectors: List[sp.Matrix]) -> int:
    if not vectors:
        return 0
    return int(sp.Matrix.hstack(*vectors).rank())


def _bracket(c, x: sp.Matrix, y: sp.Matrix) -> sp.Matrix:
    n = c.shape[0]
    out = sp.Matrix([0] * n)
    for i in range(n):
        if x[i] == 0:
            continue
        for j in range(n):
            if y[j] == 0:
                continue
            coeff = x[i] * y[j]
            if coeff == 0:
                continue
            for k in range(n):
                val = c[i, j, k]
                if val != 0:
                    out[k] += coeff * val
    return out


def _derived_subspace(c) -> List[sp.Matrix]:
    n = c.shape[0]
    cols: List[sp.Matrix] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = sp.Matrix([c[i, j, k] for k in range(n)])
            if any(v):
                cols.append(v)
    if not cols:
        return []
    return list(sp.Matrix.hstack(*cols).columnspace())


def _center_dim(c) -> int:
    n = c.shape[0]
    z = [sp.Symbol(f"z{i}") for i in range(n)]
    eqs = []
    for j in range(n):
        for k in range(n):
            eqs.append(sum(z[i] * c[i, j, k] for i in range(n)))
    A, _ = sp.linear_eq_to_matrix(eqs, z)
    return int(len(A.nullspace()))


def _lcs_dims(c, max_steps: int = 8) -> List[int]:
    n = c.shape[0]
    g_basis = [sp.Matrix([1 if r == i else 0 for r in range(n)]) for i in range(n)]
    dims = [n]
    current = g_basis
    for _ in range(max_steps):
        cols: List[sp.Matrix] = []
        for x in g_basis:
            for y in current:
                v = _bracket(c, x, y)
                if any(v):
                    cols.append(v)
        if not cols:
            dims.append(0)
            break
        current = list(sp.Matrix.hstack(*cols).columnspace())
        dims.append(len(current))
        if dims[-1] == 0:
            break
        if len(dims) > 2 and dims[-1] == dims[-2]:
            break
    return dims


def _ds_dims(c, max_steps: int = 8) -> List[int]:
    n = c.shape[0]
    current = [sp.Matrix([1 if r == i else 0 for r in range(n)]) for i in range(n)]
    dims: List[int] = []
    for _ in range(max_steps):
        cols: List[sp.Matrix] = []
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                v = _bracket(c, current[i], current[j])
                if any(v):
                    cols.append(v)
        if not cols:
            dims.append(0)
            break
        current = list(sp.Matrix.hstack(*cols).columnspace())
        dims.append(len(current))
        if dims[-1] == 0:
            break
    return dims


def _quick_hash(payload: dict) -> str:
    return hashlib.sha1(json_dumps(payload).encode("utf-8")).hexdigest()


def _to_int_list(xs) -> List[int]:
    return [int(x) for x in list(xs)]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage3_invariants.csv"
    summary_path = out_dir / "stage3_invariants_summary.json"

    count = 0
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
                "center_dim",
                "derived_dim",
                "lcs_dims",
                "ds_dims",
                "output_support_size",
                "quick_hash",
            ],
        )
        writer.writeheader()

        files = discover_npz_files(dataset_root)
        for file_idx, npz_path in enumerate(files, start=1):
            print(f"[stage3] file {file_idx}/{len(files)}: {npz_path}", flush=True)
            family = npz_path.parent.name
            batch = load_batch_file(npz_path, compute_hash=False)
            for g in iter_graphs(batch):
                count += 1
                if count % 100 == 0:
                    print(f"[stage3] processed graphs: {count}", flush=True)
                dense = dense_structure_constants(g)
                c = sp.ImmutableDenseNDimArray(dense.tolist())

                center_dim = int(_center_dim(c))
                derived = _derived_subspace(c)
                derived_dim = int(len(derived))
                lcs = _to_int_list(_lcs_dims(c))
                ds = _to_int_list(_ds_dims(c))

                output_support_size = int(len(set(int(x) for x in g.k.tolist())))
                payload = {
                    "n": g.n,
                    "nv": g.nv,
                    "nz": g.nz,
                    "center_dim": center_dim,
                    "derived_dim": derived_dim,
                    "lcs": lcs,
                    "ds": ds,
                    "output_support_size": output_support_size,
                }

                writer.writerow(
                    {
                        "graph_id": g.graph_id,
                        "family": family,
                        "source_file": npz_path.as_posix(),
                        "graph_index": g.graph_index,
                        "n": g.n,
                        "nv": g.nv,
                        "nz": g.nz,
                        "entry_count": g.ptr_end - g.ptr_start,
                        "center_dim": center_dim,
                        "derived_dim": derived_dim,
                        "lcs_dims": json.dumps(lcs, ensure_ascii=False),
                        "ds_dims": json.dumps(ds, ensure_ascii=False),
                        "output_support_size": output_support_size,
                        "quick_hash": _quick_hash(payload),
                    }
                )

                if args.max_graphs > 0 and count >= args.max_graphs:
                    break
            if args.max_graphs > 0 and count >= args.max_graphs:
                break

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "graphs_processed": count,
        "invariants_csv": csv_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
