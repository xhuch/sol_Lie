from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np


REQUIRED_KEYS = {
    "num_graphs",
    "graph_nv",
    "graph_nz",
    "graph_dim",
    "graph_ptr",
    "i",
    "j",
    "k",
    "c",
}


@dataclass
class GraphRecord:
    file_path: Path
    file_sha256: str
    file_index: int
    graph_index: int
    graph_id: str
    n: int
    nv: int
    nz: int
    ptr_start: int
    ptr_end: int
    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    c: np.ndarray
    metadata: Dict[str, object]


@dataclass
class BatchFile:
    path: Path
    sha256: str
    arrays: Dict[str, np.ndarray]


class DatasetFormatError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_npz_files(dataset_root: Path) -> List[Path]:
    files = sorted(dataset_root.glob("N*/**/*_sparse.npz"))
    return [p for p in files if p.is_file()]


def _scalar(v: np.ndarray):
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    return v


def load_batch_file(path: Path, compute_hash: bool = False) -> BatchFile:
    sha = file_sha256(path) if compute_hash else ""
    z = np.load(path, allow_pickle=True)
    arrays = {k: z[k] for k in z.files}
    return BatchFile(path=path, sha256=sha, arrays=arrays)


def validate_batch_keys(arrays: Dict[str, np.ndarray]) -> List[str]:
    missing = sorted(REQUIRED_KEYS - set(arrays.keys()))
    return missing


def batch_metadata(arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    out = {}
    for key in (
        "source_mat_file",
        "source_variable",
        "index_base",
        "antisymmetric",
        "basis_order",
        "storage_convention",
        "num_graphs",
    ):
        if key in arrays:
            out[key] = _scalar(arrays[key])
    return out


def iter_graphs(batch: BatchFile) -> Iterator[GraphRecord]:
    arrays = batch.arrays
    missing = validate_batch_keys(arrays)
    if missing:
        raise DatasetFormatError(f"{batch.path}: missing keys {missing}")

    graph_ptr = np.asarray(arrays["graph_ptr"], dtype=np.int64)
    graph_nv = np.asarray(arrays["graph_nv"], dtype=np.int64)
    graph_nz = np.asarray(arrays["graph_nz"], dtype=np.int64)
    graph_dim = np.asarray(arrays["graph_dim"], dtype=np.int64)
    i_all = np.asarray(arrays["i"], dtype=np.int64)
    j_all = np.asarray(arrays["j"], dtype=np.int64)
    k_all = np.asarray(arrays["k"], dtype=np.int64)
    c_all = np.asarray(arrays["c"])

    if graph_ptr.ndim != 1:
        raise DatasetFormatError(f"{batch.path}: graph_ptr must be 1D")
    if graph_ptr.size < 2:
        raise DatasetFormatError(f"{batch.path}: graph_ptr must have size >= 2")

    num_graphs = int(graph_ptr.size - 1)
    for arr_name, arr in (("graph_nv", graph_nv), ("graph_nz", graph_nz), ("graph_dim", graph_dim)):
        if arr.size != num_graphs:
            raise DatasetFormatError(
                f"{batch.path}: {arr_name}.size={arr.size} incompatible with num_graphs={num_graphs}"
            )

    for g in range(num_graphs):
        s = int(graph_ptr[g])
        e = int(graph_ptr[g + 1])
        rel = batch.path.relative_to(batch.path.parents[1])
        graph_id = f"{rel.as_posix()}::g{g:06d}"
        yield GraphRecord(
            file_path=batch.path,
            file_sha256=batch.sha256,
            file_index=0,
            graph_index=g,
            graph_id=graph_id,
            n=int(graph_dim[g]),
            nv=int(graph_nv[g]),
            nz=int(graph_nz[g]),
            ptr_start=s,
            ptr_end=e,
            i=i_all[s:e],
            j=j_all[s:e],
            k=k_all[s:e],
            c=c_all[s:e],
            metadata=batch_metadata(arrays),
        )


def dense_structure_constants(graph: GraphRecord) -> np.ndarray:
    n = graph.n
    c = np.zeros((n, n, n), dtype=np.int64)
    ii = graph.i.astype(np.int64, copy=False)
    jj = graph.j.astype(np.int64, copy=False)
    kk = graph.k.astype(np.int64, copy=False)
    vv = graph.c.astype(np.int64, copy=False)
    c[ii, jj, kk] += vv
    c[jj, ii, kk] -= vv
    return c


def json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)
