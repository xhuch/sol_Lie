from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np

from dataset_batch import discover_npz_files, load_batch_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: validate batch NPZ dataset integrity.")
    p.add_argument("--dataset-root", default="datasets", help="Root folder that contains N6/N7/N8/N9")
    p.add_argument("--out-dir", default="artifacts/niliso", help="Output directory")
    p.add_argument("--hash-files", action="store_true", help="Compute SHA256 for each NPZ file (slower).")
    return p.parse_args()


def _check_file(batch_path: Path, compute_hash: bool = False) -> List[dict]:
    rows: List[dict] = []
    batch = load_batch_file(batch_path, compute_hash=compute_hash)
    a = batch.arrays
    missing = sorted({"num_graphs", "graph_nv", "graph_nz", "graph_dim", "graph_ptr", "i", "j", "k", "c"} - set(a.keys()))
    if missing:
        rows.append(
            {
                "level": "file",
                "source_file": batch_path.as_posix(),
                "graph_index": -1,
                "code": "missing_keys",
                "ok": 0,
                "detail": f"missing={missing}",
            }
        )
        return rows

    graph_ptr = np.asarray(a["graph_ptr"], dtype=np.int64)
    graph_nv = np.asarray(a["graph_nv"], dtype=np.int64)
    graph_nz = np.asarray(a["graph_nz"], dtype=np.int64)
    graph_dim = np.asarray(a["graph_dim"], dtype=np.int64)
    i_all = np.asarray(a["i"], dtype=np.int64)
    j_all = np.asarray(a["j"], dtype=np.int64)
    k_all = np.asarray(a["k"], dtype=np.int64)

    num_graphs = int(graph_ptr.size - 1) if graph_ptr.ndim == 1 and graph_ptr.size >= 2 else -1
    decl_num_graphs = int(a["num_graphs"].item()) if np.asarray(a["num_graphs"]).shape == () else None

    def add_file(code: str, ok: bool, detail: str) -> None:
        rows.append(
            {
                "level": "file",
                "source_file": batch_path.as_posix(),
                "graph_index": -1,
                "code": code,
                "ok": int(ok),
                "detail": detail,
            }
        )

    add_file("num_graphs_decl_match", decl_num_graphs == num_graphs, f"decl={decl_num_graphs}, ptr_count={num_graphs}")

    if num_graphs < 0:
        add_file("graph_ptr_shape", False, f"shape={graph_ptr.shape}")
        return rows

    add_file("graph_ptr_monotone", bool(np.all(graph_ptr[1:] >= graph_ptr[:-1])), "")
    add_file("graph_ptr_start_zero", int(graph_ptr[0]) == 0, f"start={int(graph_ptr[0])}")
    add_file("graph_ptr_end_matches_entries", int(graph_ptr[-1]) == int(i_all.size), f"end={int(graph_ptr[-1])}, entries={int(i_all.size)}")

    for arr_name, arr in (("graph_nv", graph_nv), ("graph_nz", graph_nz), ("graph_dim", graph_dim)):
        add_file(f"{arr_name}_size_match", int(arr.size) == num_graphs, f"size={arr.size}, expected={num_graphs}")

    index_base = int(a.get("index_base", np.array(0)).item()) if np.asarray(a.get("index_base", np.array(0))).shape == () else None
    if index_base is not None:
        add_file("index_base_zero", index_base == 0, f"index_base={index_base}")
    antisym = bool(a.get("antisymmetric", np.array(True)).item()) if np.asarray(a.get("antisymmetric", np.array(True))).shape == () else None
    if antisym is not None:
        add_file("antisymmetric_true", antisym is True, f"antisymmetric={antisym}")

    for g in range(num_graphs):
        s = int(graph_ptr[g])
        e = int(graph_ptr[g + 1])
        n = int(graph_dim[g])
        nv = int(graph_nv[g])
        nz = int(graph_nz[g])
        gi = i_all[s:e]
        gj = j_all[s:e]
        gk = k_all[s:e]

        def add_graph(code: str, ok: bool, detail: str = "") -> None:
            rows.append(
                {
                    "level": "graph",
                    "source_file": batch_path.as_posix(),
                    "graph_index": g,
                    "code": code,
                    "ok": int(ok),
                    "detail": detail,
                }
            )

        add_graph("nv_plus_nz_eq_dim", nv + nz == n, f"nv={nv}, nz={nz}, dim={n}")
        add_graph("indices_i_lt_j", bool(np.all(gi < gj)), "")
        add_graph("indices_non_negative", bool(np.all(gi >= 0) and np.all(gj >= 0) and np.all(gk >= 0)), "")
        add_graph("indices_lt_dim", bool(np.all(gi < n) and np.all(gj < n) and np.all(gk < n)), f"dim={n}")

    return rows


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    files = discover_npz_files(dataset_root)
    for idx, npz_path in enumerate(files, start=1):
        print(f"[stage2] validating {idx}/{len(files)}: {npz_path}", flush=True)
        rows.extend(_check_file(npz_path, compute_hash=args.hash_files))

    csv_path = out_dir / "stage2_validation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "source_file", "graph_index", "code", "ok", "detail"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    failed = sum(1 for r in rows if int(r["ok"]) == 0)
    summary = {
        "dataset_root": dataset_root.as_posix(),
        "checks_total": total,
        "checks_failed": failed,
        "checks_passed": total - failed,
        "validation_csv": csv_path.as_posix(),
    }
    summary_path = out_dir / "stage2_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
