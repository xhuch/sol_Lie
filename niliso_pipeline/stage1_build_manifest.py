from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from dataset_batch import discover_npz_files, iter_graphs, load_batch_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: build nilpotent dataset manifest from batch NPZ files.")
    p.add_argument("--dataset-root", default="datasets", help="Root folder that contains N6/N7/N8/N9")
    p.add_argument("--out-dir", default="artifacts/niliso", help="Output directory")
    p.add_argument("--hash-files", action="store_true", help="Compute SHA256 for each NPZ file (slower).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_npz_files(dataset_root)
    manifest_path = out_dir / "stage1_manifest.csv"
    summary_path = out_dir / "stage1_summary.json"

    total_graphs = 0
    family_counts = {}

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_id",
                "family",
                "source_file",
                "source_file_sha256",
                "graph_index",
                "n",
                "nv",
                "nz",
                "ptr_start",
                "ptr_end",
                "entry_count",
                "source_variable",
                "source_mat_file",
            ],
        )
        writer.writeheader()

        for idx, npz_path in enumerate(files, start=1):
            print(f"[stage1] loading {idx}/{len(files)}: {npz_path}", flush=True)
            batch = load_batch_file(npz_path, compute_hash=args.hash_files)
            family = npz_path.parent.name
            family_counts[family] = family_counts.get(family, 0) + 1
            for graph in iter_graphs(batch):
                total_graphs += 1
                writer.writerow(
                    {
                        "graph_id": graph.graph_id,
                        "family": family,
                        "source_file": npz_path.as_posix(),
                        "source_file_sha256": graph.file_sha256,
                        "graph_index": graph.graph_index,
                        "n": graph.n,
                        "nv": graph.nv,
                        "nz": graph.nz,
                        "ptr_start": graph.ptr_start,
                        "ptr_end": graph.ptr_end,
                        "entry_count": graph.ptr_end - graph.ptr_start,
                        "source_variable": graph.metadata.get("source_variable", ""),
                        "source_mat_file": graph.metadata.get("source_mat_file", ""),
                    }
                )

    summary = {
        "dataset_root": dataset_root.as_posix(),
        "npz_file_count": len(files),
        "graph_count": total_graphs,
        "family_file_counts": dict(sorted(family_counts.items())),
        "manifest_csv": manifest_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
