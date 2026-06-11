import argparse
import concurrent.futures
import glob
import hashlib
import importlib.util
import itertools
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp


Vector = sp.Matrix
CACHE_SCHEMA_VERSION = 2
CACHE_ALGO_TAG = "post5-opt6"


def load_module(module_path: Path):
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "build_structure_constants"):
        raise AttributeError(f"{module_path} does not define build_structure_constants(n)")
    return module


def bracket_vector(c, i: int, j: int) -> Vector:
    n = len(c)
    return sp.Matrix([c[i][j][k] for k in range(n)])


def derived_subspace(c) -> List[Vector]:
    n = len(c)
    cols: List[Vector] = []
    for i in range(n):
        for j in range(i + 1, n):
            v = bracket_vector(c, i, j)
            if any(v):
                cols.append(v)
    if not cols:
        return []
    return sp.Matrix.hstack(*cols).columnspace()


def center_subspace(c) -> List[Vector]:
    n = len(c)
    z = [sp.Symbol(f"z{i+1}") for i in range(n)]
    eqs = []
    for j in range(n):
        for k in range(n):
            eqs.append(sum(z[i] * c[i][j][k] for i in range(n)))
    A, _ = sp.linear_eq_to_matrix(eqs, z)
    return A.nullspace()


def derivation_dimension(c) -> int:
    n = len(c)
    D_vars = [sp.Symbol(f"d{r+1}_{s+1}") for r in range(n) for s in range(n)]
    eqs = []
    for i in range(n):
        for j in range(n):
            if not any(c[i][j][k] != 0 for k in range(n)):
                continue
            for k in range(n):
                left = sum(c[i][j][m] * D_vars[k * n + m] for m in range(n))
                right1 = sum(D_vars[p * n + i] * c[p][j][k] for p in range(n))
                right2 = sum(D_vars[q * n + j] * c[i][q][k] for q in range(n))
                eqs.append(sp.expand(left - right1 - right2))
    if not eqs:
        return n * n
    A, _ = sp.linear_eq_to_matrix(eqs, D_vars)
    return int(len(A.nullspace()))


def extension_weight_multiset(ca, m: int, n: int) -> List[Tuple[str, ...]]:
    pre = precompute_nil_extension_weight_strings(ca, m, n)
    return sorted(pre)


def precompute_nil_extension_weight_strings(ca, m: int, n: int) -> List[Tuple[str, ...]]:
    ext_idx = list(range(m, n))
    out = []
    for i in range(m):
        out.append(tuple(str(sp.simplify(ca[i][t][i])) for t in ext_idx))
    return out


def precompute_extension_action_stats(ca, m: int, n: int) -> List[Tuple[str, str]]:
    """Build extension action matrices once and reuse rank/trace summaries."""
    stats: List[Tuple[str, str]] = []
    nil_idx = range(m)
    for t_idx in range(m, n):
        rows = [[ca[j][t_idx][i] for j in nil_idx] for i in nil_idx]
        action = sp.Matrix(rows)
        stats.append((str(sp.simplify(action.trace())), str(int(action.rank()))))
    return stats


def lower_central_series_dims(c, max_steps: int = 8) -> List[int]:
    n = len(c)
    g_basis = [sp.Matrix([1 if r == i else 0 for r in range(n)]) for i in range(n)]
    dims = [n]
    current = g_basis
    for _ in range(max_steps):
        cols: List[Vector] = []
        for x in g_basis:
            for y in current:
                v = bracket_of_vectors(c, x, y)
                if any(v):
                    cols.append(v)
        if not cols:
            dims.append(0)
            break
        current = sp.Matrix.hstack(*cols).columnspace()
        dims.append(len(current))
        if dims[-1] == dims[-2]:
            break
    return dims


def derived_series_dims(c, max_steps: int = 8) -> List[int]:
    n = len(c)
    current = [sp.Matrix([1 if r == i else 0 for r in range(n)]) for i in range(n)]
    dims = []
    for _ in range(max_steps):
        cols: List[Vector] = []
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                v = bracket_of_vectors(c, current[i], current[j])
                if any(v):
                    cols.append(v)
        if not cols:
            dims.append(0)
            break
        current = sp.Matrix.hstack(*cols).columnspace()
        dims.append(len(current))
        if dims[-1] == 0:
            break
    return dims


def bracket_of_vectors(c, x: Vector, y: Vector) -> Vector:
    n = len(c)
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
            cij = c[i][j]
            for k in range(n):
                if cij[k] != 0:
                    out[k] += coeff * cij[k]
    return out


def complete_basis(subspace_basis: Sequence[Vector], n: int) -> List[Vector]:
    basis = list(subspace_basis)
    if basis:
        M = sp.Matrix.hstack(*basis)
    else:
        M = sp.zeros(n, 0)
    for i in range(n):
        e = sp.Matrix([1 if r == i else 0 for r in range(n)])
        trial = M.row_join(e)
        if trial.rank() > M.rank():
            basis.append(e)
            M = trial
        if len(basis) == n:
            break
    if len(basis) != n:
        raise RuntimeError("Failed to complete basis")
    return basis


def change_structure_constants(c, basis: Sequence[Vector]):
    n = len(c)
    P = sp.Matrix.hstack(*basis)
    Pinv = P.inv()
    cp = [[[sp.Rational(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            bi = P[:, i]
            bj = P[:, j]
            bracket_orig = bracket_of_vectors(c, bi, bj)
            coords = Pinv * bracket_orig
            for k in range(n):
                cp[i][j][k] = sp.simplify(coords[k])
    return cp, P


@dataclass
class AlgebraReport:
    n: int
    center_dim: int
    derived_dim: int
    lower_central_dims: List[int]
    derived_series_dims: List[int]
    nilradical_candidate_dim: int
    extension_dim: int
    adapted_structure_constants: list
    n_basis_dim: int
    extension_order_adapted_1based: List[int]
    nil_order_adapted_1based: List[int]
    weight_blocks: List[Dict[str, object]]
    nil_weight_labels: List[Tuple[str, ...]]
    extension_labels: List[Tuple[str, ...]]
    nil_kernel_labels: List[Tuple[str, ...]]
    nil_vertex_color_labels: List[Tuple[str, ...]]
    nil_colored_hyperedges: List[Tuple[int, int, Tuple[str, ...]]]
    derivation_dim: Optional[int]
    extension_weight_multiset: List[Tuple[str, ...]]
    adapted_profile_context: Optional[Dict[str, object]] = None


def bracket_output_indices(c) -> List[int]:
    n = len(c)
    idx = set()
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(n):
                if c[i][j][k] != 0:
                    idx.add(k)
    return sorted(idx)


def permute_structure_constants(c, order: Sequence[int]):
    n = len(c)
    cp = [[[sp.Rational(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for i_new in range(n):
        i_old = order[i_new]
        for j_new in range(n):
            j_old = order[j_new]
            for k_new in range(n):
                k_old = order[k_new]
                cp[i_new][j_new][k_new] = c[i_old][j_old][k_old]
    return cp


def standard_center_indices(ca, m: int) -> List[int]:
    center_idx = []
    for i in range(m):
        is_center = True
        for j in range(m):
            if any(ca[i][j][k] != 0 for k in range(m)):
                is_center = False
                break
        if is_center:
            center_idx.append(i)
    return center_idx


def build_adapted_profile_context(ca, m: int, n: int) -> Dict[str, object]:
    center_idx = standard_center_indices(ca, m)
    center_set = set(center_idx)
    v_idx = [i for i in range(m) if i not in center_set]
    ext_idx = list(range(m, n))
    action_mats = [sp.Matrix([[ca[j][t][i] for j in range(m)] for i in range(m)]) for t in ext_idx]
    action_ranks = [int(A.rank()) for A in action_mats]
    action_traces = [sp.simplify(A.trace()) for A in action_mats]

    fast_2step = None
    if center_idx:
        fast_ok = True
        for i in range(m):
            for j in range(i + 1, m):
                for k in range(m):
                    if k not in center_set and ca[i][j][k] != 0:
                        fast_ok = False
                        break
                if not fast_ok:
                    break
            if not fast_ok:
                break

        if fast_ok:
            zdim = len(center_idx)
            vdim = len(v_idx)
            component_mats = [sp.zeros(vdim, vdim) for _ in range(zdim)]
            all_coords_cols = []
            for a, i in enumerate(v_idx):
                for b in range(a + 1, vdim):
                    j = v_idx[b]
                    coords = [ca[i][j][z] for z in center_idx]
                    all_coords_cols.append(sp.Matrix(coords))
                    for r, z in enumerate(center_idx):
                        val = ca[i][j][z]
                        component_mats[r][a, b] = val
                        component_mats[r][b, a] = -val

            fast_2step = None
            fast_2step_raw = {
                "Z_dim": zdim,
                "V_dim": vdim,
                "pair_count": vdim * (vdim - 1) // 2,
                "component_mats": component_mats,
                "all_coords_cols": all_coords_cols,
            }
        else:
            fast_2step_raw = None
    else:
        fast_2step_raw = None

    return {
        "center_idx": center_idx,
        "center_set": center_set,
        "v_idx": v_idx,
        "ext_idx": ext_idx,
        "action_mats": action_mats,
        "action_ranks": action_ranks,
        "action_traces": action_traces,
        "matrix_group_signature_cached": None,
        "fast_2step": fast_2step,
        "fast_2step_raw": fast_2step_raw,
    }


def get_profile_context(rep: AlgebraReport) -> Dict[str, object]:
    ctx = rep.adapted_profile_context
    if not isinstance(ctx, dict):
        ctx = build_adapted_profile_context(rep.adapted_structure_constants, rep.n_basis_dim, rep.n)
        rep.adapted_profile_context = ctx
    return ctx


def _ctx_get_list(ctx: Optional[Dict[str, object]], key: str) -> List[Any]:
    if not ctx:
        return []
    val = ctx.get(key)
    return val if isinstance(val, list) else []


def _rat_modp(x, p: int) -> Optional[int]:
    try:
        r = sp.Rational(x)
    except Exception:
        return None
    num = int(r.p) % p
    den = int(r.q) % p
    if den == 0:
        return None
    return (num * pow(den, -1, p)) % p


def _matrix_to_modp_entries(M: sp.MatrixBase, p: int) -> Optional[List[List[int]]]:
    rows, cols = M.shape
    out = [[0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            v = _rat_modp(M[i, j], p)
            if v is None:
                return None
            out[i][j] = v
    return out


def _rank_modp_entries(entries: List[List[int]], p: int) -> int:
    if not entries:
        return 0
    a = [row[:] for row in entries]
    m = len(a)
    n = len(a[0]) if m else 0
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if a[r][col] % p != 0:
                pivot = r
                break
        if pivot is None:
            continue
        a[row], a[pivot] = a[pivot], a[row]
        inv = pow(a[row][col] % p, -1, p)
        for c in range(col, n):
            a[row][c] = (a[row][c] * inv) % p
        for r in range(m):
            if r == row:
                continue
            factor = a[r][col] % p
            if factor == 0:
                continue
            for c in range(col, n):
                a[r][c] = (a[r][c] - factor * a[row][c]) % p
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def _rank_sympy_matrix_modp(M: sp.MatrixBase, p: int) -> Optional[int]:
    entries = _matrix_to_modp_entries(M, p)
    if entries is None:
        return None
    return _rank_modp_entries(entries, p)


def _trace_modp(M: sp.MatrixBase, p: int) -> Optional[int]:
    t = sp.simplify(M.trace())
    return _rat_modp(t, p)


def bracket_tensor_profile_modp(ca, m: int, primes: Sequence[int], ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    ctx = ctx or build_adapted_profile_context(ca, m, len(ca))
    raw = ctx.get("fast_2step_raw")
    if not isinstance(raw, dict):
        return {
            "available": False,
            "reason": "fast_2step_not_applicable",
            "primes": list(primes),
            "signature": None,
            "by_prime": [],
        }

    component_mats = raw.get("component_mats")
    all_coords_cols = raw.get("all_coords_cols")
    if not isinstance(component_mats, list) or not isinstance(all_coords_cols, list):
        return {
            "available": False,
            "reason": "missing_fast_2step_components",
            "primes": list(primes),
            "signature": None,
            "by_prime": [],
        }

    by_prime = []
    for p in primes:
        p = int(p)
        comp_ranks = []
        ok = True
        for M in component_mats:
            if not isinstance(M, sp.MatrixBase):
                ok = False
                break
            rk = _rank_sympy_matrix_modp(M, p)
            if rk is None:
                ok = False
                break
            comp_ranks.append(int(rk))
        if not ok:
            continue

        if all_coords_cols:
            stacked = sp.Matrix.hstack(*all_coords_cols)
            map_rank = _rank_sympy_matrix_modp(stacked, p)
            if map_rank is None:
                continue
            map_rank = int(map_rank)
        else:
            map_rank = 0

        by_prime.append(
            {
                "p": p,
                "map_rank": map_rank,
                "component_ranks": comp_ranks,
            }
        )

    signature = [(b["p"], b["map_rank"], tuple(b["component_ranks"])) for b in by_prime]
    return {
        "available": bool(by_prime),
        "reason": None if by_prime else "no_usable_prime",
        "primes": list(primes),
        "signature": signature,
        "by_prime": by_prime,
    }


def extension_action_profile_modp(ca, m: int, n: int, primes: Sequence[int], ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    ctx = ctx or build_adapted_profile_context(ca, m, n)
    action_mats = [mat for mat in _ctx_get_list(ctx, "action_mats") if isinstance(mat, sp.MatrixBase)]
    if len(action_mats) != (n - m):
        action_mats = [sp.Matrix([[ca[j][t_idx][i] for j in range(m)] for i in range(m)]) for t_idx in range(m, n)]

    by_prime = []
    for p in primes:
        p = int(p)
        actions_sig = []
        ok = True
        for A in action_mats:
            rk = _rank_sympy_matrix_modp(A, p)
            tr = _trace_modp(A, p)
            if rk is None or tr is None:
                ok = False
                break
            actions_sig.append((int(rk), int(tr)))
        if not ok:
            continue

        joint_sig = None
        if len(action_mats) == 2:
            A1 = action_mats[0]
            A2 = action_mats[1]
            stacked = A1.col_join(A2)
            hstacked = sp.Matrix.hstack(A1, A2)
            comm = A1 * A2 - A2 * A1
            rk_stacked = _rank_sympy_matrix_modp(stacked, p)
            rk_hstacked = _rank_sympy_matrix_modp(hstacked, p)
            rk_comm = _rank_sympy_matrix_modp(comm, p)
            if rk_stacked is None or rk_hstacked is None or rk_comm is None:
                ok = False
            else:
                common_kernel_dim = int(m - rk_stacked)
                image_sum_dim = int(rk_hstacked)
                combo = []
                for a, b in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)]:
                    M = a * A1 + b * A2
                    rk = _rank_sympy_matrix_modp(M, p)
                    tr = _trace_modp(M, p)
                    if rk is None or tr is None:
                        ok = False
                        break
                    combo.append((a, b, int(rk), int(tr)))
                if ok:
                    joint_sig = {
                        "common_kernel_dim": common_kernel_dim,
                        "image_sum_dim": image_sum_dim,
                        "commutator_rank": int(rk_comm),
                        "combo_rank_trace_signature": combo,
                    }
        if not ok:
            continue

        by_prime.append(
            {
                "p": p,
                "actions_signature": sorted(actions_sig),
                "joint_plane": joint_sig,
            }
        )

    signature = [(b["p"], tuple(b["actions_signature"]), b.get("joint_plane")) for b in by_prime]
    return {
        "available": bool(by_prime),
        "reason": None if by_prime else "no_usable_prime",
        "primes": list(primes),
        "signature": signature,
        "by_prime": by_prime,
    }


def extension_sort_key(ca, m: int, t_idx: int):
    A = sp.Matrix([[ca[j][t_idx][i] for j in range(m)] for i in range(m)])
    diag_multiset = tuple(sorted(str(sp.simplify(A[i, i])) for i in range(m)))
    non_diag = 0
    for i in range(m):
        for j in range(m):
            if i != j and A[i, j] != 0:
                non_diag += 1
    return (str(sp.simplify(A.trace())), int(A.rank()), non_diag, diag_multiset)


def nil_support_profile(ca, m: int, i: int) -> Tuple[Tuple[int, ...], ...]:
    profile = []
    for j in range(m):
        if i == j:
            continue
        supp = tuple(k + 1 for k in range(m) if ca[i][j][k] != 0)
        if supp:
            profile.append(supp)
    return tuple(sorted(profile))


def joint_weight_normalize(ca, m: int, n: int):
    ext_positions = list(range(m, n))
    ext_positions = sorted(ext_positions, key=lambda t: extension_sort_key(ca, m, t))
    center_idx = set(standard_center_indices(ca, m))
    weight_by_index = precompute_nil_extension_weight_strings(ca, m, n)

    def nil_key(i: int):
        weight = tuple(weight_by_index[i][t - m] for t in ext_positions)
        ext_support = tuple(
            tuple(k + 1 for k in range(m) if k != i and ca[i][t][k] != 0)
            for t in ext_positions
        )
        return (
            1 if i in center_idx else 0,
            weight,
            ext_support,
            nil_support_profile(ca, m, i),
        )

    nil_positions = list(range(m))
    nil_positions = sorted(nil_positions, key=nil_key)
    order = nil_positions + ext_positions
    ca2 = permute_structure_constants(ca, order)

    weight_blocks = []
    current_key = None
    current_block = []
    for pos in nil_positions:
        key = tuple(weight_by_index[pos][t - m] for t in ext_positions)
        if key != current_key:
            if current_block:
                weight_blocks.append({
                    "weight": list(current_key),
                    "size": len(current_block),
                    "original_indices_1based": [x + 1 for x in current_block],
                })
            current_key = key
            current_block = [pos]
        else:
            current_block.append(pos)
    if current_block:
        weight_blocks.append({
            "weight": list(current_key),
            "size": len(current_block),
            "original_indices_1based": [x + 1 for x in current_block],
        })

    return ca2, [t + 1 for t in ext_positions], [i + 1 for i in nil_positions], weight_blocks


def matrix_group_signature(ca, m: int, ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    ctx = ctx or build_adapted_profile_context(ca, m, len(ca))
    cached = ctx.get("matrix_group_signature_cached") if isinstance(ctx, dict) else None
    if isinstance(cached, dict):
        return cached

    center_idx = [int(x) for x in _ctx_get_list(ctx, "center_idx")]
    v_idx = [int(x) for x in _ctx_get_list(ctx, "v_idx")]
    component_signatures = []
    for z in center_idx:
        M = sp.zeros(len(v_idx), len(v_idx))
        support_pairs = []
        for a, i in enumerate(v_idx):
            for b, j in enumerate(v_idx):
                M[a, b] = sp.simplify(ca[i][j][z])
                if a < b and M[a, b] != 0:
                    support_pairs.append((a + 1, b + 1))
        component_signatures.append(
            {
                "rank": int(M.rank()),
                "upper_nnz": len(support_pairs),
                "support_pairs": support_pairs,
            }
        )
    component_signatures = sorted(
        component_signatures,
        key=lambda item: (item["rank"], item["upper_nnz"], tuple(item["support_pairs"])),
    )
    weak_signature = sorted((item["rank"], item["upper_nnz"]) for item in component_signatures)
    result = {
        "center_indices_1based": [i + 1 for i in center_idx],
        "V_indices_1based": [i + 1 for i in v_idx],
        "component_count": len(component_signatures),
        "weak_signature": weak_signature,
        "component_signatures": component_signatures,
    }
    if isinstance(ctx, dict):
        ctx["matrix_group_signature_cached"] = result
    return result


def nil_kernel_block_labels(ca, m: int) -> List[Tuple[str, ...]]:
    center_idx = standard_center_indices(ca, m)
    center_set = set(center_idx)
    v_idx = [i for i in range(m) if i not in center_set]
    v_pos = {v: p for p, v in enumerate(v_idx)}

    # Per-center component descriptors.
    center_desc = {}
    v_degree = {v: 0 for v in v_idx}
    for z in center_idx:
        support_pairs = []
        for a, i in enumerate(v_idx):
            for b, j in enumerate(v_idx):
                if a < b and ca[i][j][z] != 0:
                    support_pairs.append((a, b))
                    v_degree[i] += 1
                    v_degree[j] += 1
        center_desc[z] = ("Z", str(len(support_pairs)))

    labels: List[Tuple[str, ...]] = []
    for i in range(m):
        if i in center_set:
            labels.append(tuple(center_desc[i]))
        else:
            labels.append(("V", str(v_degree[i])))
    return labels


def nil_vertex_color_labels(ca, m: int, n: int) -> List[Tuple[str, ...]]:
    center_idx = standard_center_indices(ca, m)
    center_set = set(center_idx)
    weight_by_index = precompute_nil_extension_weight_strings(ca, m, n)

    # local degree profile in the nil kernel
    deg = [0] * m
    for i in range(m):
        for j in range(i + 1, m):
            if any(ca[i][j][k] != 0 for k in range(m)):
                deg[i] += 1
                deg[j] += 1

    labels: List[Tuple[str, ...]] = []
    for i in range(m):
        role = "Z" if i in center_set else "V"
        weight = tuple(weight_by_index[i])
        out_support = tuple(str(k + 1) for k in range(m) if any(ca[i][j][k] != 0 for j in range(m)))
        labels.append((role, str(deg[i])) + weight + out_support)
    return labels


def nil_colored_hyperedges(ca, m: int, n: int) -> List[Tuple[int, int, Tuple[str, ...]]]:
    """
    Encode bracket tensor on nilradical as colored 2-hyperedges on nil basis indices.
    Edge color = sorted list of (center_index, coeff) outputs + extension-action color of output center.
    """
    weight_by_index = precompute_nil_extension_weight_strings(ca, m, n)
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            comps = []
            for k in range(m):
                coeff = sp.simplify(ca[i][j][k])
                if coeff != 0:
                    k_color = tuple(weight_by_index[k])
                    comps.append((k + 1, str(coeff), k_color))
            if comps:
                comps = sorted(comps, key=lambda x: (x[0], x[1], x[2]))
                flat = []
                for kk, cc, kc in comps:
                    flat.extend([str(kk), cc, "|".join(kc)])
                edges.append((i, j, tuple(flat)))
    return edges


def wl_refine_labels(
    n_vertices: int,
    labels: Sequence[Tuple[str, ...]],
    edges: Sequence[Tuple[int, int, Tuple[str, ...]]],
    rounds: int = 6,
) -> List[Tuple[str, ...]]:
    cur = [tuple(x) for x in labels]
    incident = [[] for _ in range(n_vertices)]
    for i, j, ecolor in edges:
        incident[i].append((j, ecolor))
        incident[j].append((i, ecolor))
    for _ in range(rounds):
        nxt = []
        for v in range(n_vertices):
            neigh_sig = sorted((cur[u], ecolor) for (u, ecolor) in incident[v])
            nxt.append(("WL",) + cur[v] + (str(neigh_sig),))
        if nxt == cur:
            break
        cur = nxt
    return cur


def invariant_hash_bundle(rep: AlgebraReport, bt: Dict[str, object], ext: Dict[str, object], mg: Dict[str, object]) -> str:
    sig = {
        "n": rep.n,
        "center_dim": rep.center_dim,
        "derived_dim": rep.derived_dim,
        "ds": rep.derived_series_dims,
        "lcs": rep.lower_central_dims,
        "nil_dim": rep.nilradical_candidate_dim,
        "ext_dim": rep.extension_dim,
        "weight_block_sig": weight_block_signature(rep),
        "bt": {
            "Z_dim": bt.get("Z_dim"),
            "map_rank": bt.get("map_rank"),
            "component_ranks": bt.get("component_ranks"),
        },
        "ext": sorted((a.get("rank"), str(a.get("trace", 0))) for a in ext.get("actions", [])),
        "mg_weak": mg.get("weak_signature", []),
    }
    return json.dumps(to_jsonable(sig), sort_keys=True, ensure_ascii=False)


def light_invariant_hash(rep: AlgebraReport) -> str:
    sig = {
        "n": rep.n,
        "center_dim": rep.center_dim,
        "derived_dim": rep.derived_dim,
        "ds": rep.derived_series_dims,
        "lcs": rep.lower_central_dims,
        "nil_dim": rep.nilradical_candidate_dim,
        "ext_dim": rep.extension_dim,
        "weight_block_sig": weight_block_signature(rep),
        "extension_weight_multiset": rep.extension_weight_multiset,
    }
    return json.dumps(to_jsonable(sig), sort_keys=True, ensure_ascii=False)


def build_roi_and_suggestions(
    prefilter_reject_rate: Optional[float],
    profiles_per_pair_ms: Optional[float],
    search_tree_reduction: Optional[float],
) -> Dict[str, object]:
    suggestions: List[str] = []
    if prefilter_reject_rate is not None:
        if prefilter_reject_rate < 0.2:
            suggestions.append("Low prefilter reject rate: strengthen secondary signatures or enable deeper bucket refinement threshold sooner.")
        elif prefilter_reject_rate > 0.8:
            suggestions.append("High prefilter reject rate: keep lightweight front filters and avoid expensive deep signatures globally.")
    if profiles_per_pair_ms is not None and profiles_per_pair_ms > 700:
        suggestions.append("High profile cost per pair: reduce matrix-pencil basis cap or trigger deep signatures only for larger collision buckets.")
    if search_tree_reduction is not None and search_tree_reduction < 0.95:
        suggestions.append("Search-tree reduction is modest: keep failure-memory pruning enabled and consider stronger local color constraints.")
    if not suggestions:
        suggestions.append("Current configuration is balanced; keep defaults unless workload distribution changes.")

    return {
        "prefilter_reject_rate": prefilter_reject_rate,
        "profiles_per_pair_ms": profiles_per_pair_ms,
        "search_tree_reduction": search_tree_reduction,
        "next_run_suggestions": suggestions,
    }


def _cli_quote(value: object) -> str:
    s = str(value)
    if any(ch in s for ch in (' ', '\t', '"')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


def derive_next_run_overrides(args_dict: Dict[str, object], roi: Dict[str, object], batch_mode: bool) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    notes: List[str] = []

    prefilter_reject_rate = roi.get("prefilter_reject_rate")
    profiles_per_pair_ms = roi.get("profiles_per_pair_ms")
    search_tree_reduction = roi.get("search_tree_reduction")

    bucket_deepen_min_size = int(args_dict.get("bucket_deepen_min_size", 3))
    pencil_basis_cap = int(args_dict.get("pencil_basis_cap", 3))
    pencil_basis_cap_deep = int(args_dict.get("pencil_basis_cap_deep", 6))
    profile_modp_mode = str(args_dict.get("profile_modp_mode", "auto"))

    if isinstance(prefilter_reject_rate, (int, float)):
        if prefilter_reject_rate < 0.2:
            if batch_mode:
                target_bucket_deepen = max(2, bucket_deepen_min_size - 1)
                if target_bucket_deepen != bucket_deepen_min_size:
                    overrides["bucket_deepen_min_size"] = target_bucket_deepen
                    notes.append("Low prefilter reject rate: start deepening on slightly smaller buckets.")
        elif prefilter_reject_rate > 0.8:
            if batch_mode:
                target_bucket_deepen = max(bucket_deepen_min_size, 4)
                if target_bucket_deepen != bucket_deepen_min_size:
                    overrides["bucket_deepen_min_size"] = target_bucket_deepen
                    notes.append("High prefilter reject rate: delay deepening to larger buckets to save profile cost.")

    if isinstance(profiles_per_pair_ms, (int, float)) and profiles_per_pair_ms > 700:
        target_pencil_cap = max(2, pencil_basis_cap - 1)
        if target_pencil_cap != pencil_basis_cap:
            overrides["pencil_basis_cap"] = target_pencil_cap
            notes.append("High profile cost: lower default matrix-pencil basis cap.")
        target_pencil_cap_deep = max(4, pencil_basis_cap_deep - 1)
        if target_pencil_cap_deep != pencil_basis_cap_deep:
            overrides["pencil_basis_cap_deep"] = target_pencil_cap_deep
            notes.append("High profile cost: lower deep matrix-pencil basis cap slightly.")

    if isinstance(search_tree_reduction, (int, float)) and search_tree_reduction < 0.95:
        if profile_modp_mode == "off":
            overrides["profile_modp_mode"] = "auto"
            notes.append("Search-tree reduction modest: enable mod-p profile prefilter in auto mode.")

    return {
        "overrides": overrides,
        "notes": notes,
    }


def _build_next_run_command(args_dict: Dict[str, object], out_path: Path, overrides: Dict[str, object]) -> str:
    command: List[str] = ["python", _cli_quote(Path(__file__).name)]
    batch_mode = bool(args_dict.get("batch_glob"))
    if batch_mode:
        command.extend(["--batch-glob", _cli_quote(args_dict["batch_glob"])])
    else:
        command.extend(["--alg1", _cli_quote(args_dict["alg1"])])
        command.extend(["--alg2", _cli_quote(args_dict["alg2"])])

    n_value = int(args_dict.get("n", 23))
    command.extend(["--n", str(n_value)])

    next_out = out_path.with_name(f"{out_path.stem}_next_run{out_path.suffix or '.json'}")
    command.extend(["--out", _cli_quote(str(next_out))])

    tunable_keys = [
        "profile_modp_mode",
        "profile_modp_primes",
        "modp_primes",
        "pencil_basis_cap",
        "pencil_basis_cap_deep",
        "bucket_deepen_min_size",
        "bucket_thread_workers",
        "batch_process_workers",
        "pair_parallel_workers",
    ]
    for key in tunable_keys:
        value = overrides.get(key, args_dict.get(key))
        if value is None:
            continue
        command.extend([f"--{key.replace('_', '-')}", _cli_quote(value)])
    return " ".join(command)


def write_next_run_templates(args_dict: Dict[str, object], output: Dict[str, object], out_path: Path) -> Optional[Dict[str, object]]:
    roi = output.get("roi")
    if not isinstance(roi, dict):
        return None

    batch_mode = bool(args_dict.get("batch_glob"))
    suggestion_block = derive_next_run_overrides(args_dict, roi, batch_mode)
    overrides = suggestion_block["overrides"]
    notes = suggestion_block["notes"]

    command_line = _build_next_run_command(args_dict, out_path, overrides)
    ps1_path = out_path.with_name(f"{out_path.stem}_next_run_template.ps1")
    ps1_lines = [
        "# Auto-generated from ROI metrics in the latest report.",
        "# Edit parameters below if workload assumptions change.",
        command_line,
        "",
    ]
    ps1_path.write_text("\n".join(ps1_lines), encoding="utf-8")

    suggestions_path = out_path.with_name(f"{out_path.stem}_next_run_suggestions.json")
    suggestions_payload = {
        "generated_from_report": str(out_path),
        "mode": "batch" if batch_mode else "pair",
        "roi": roi,
        "recommended_overrides": overrides,
        "auto_notes": notes,
        "generated_at_epoch_seconds": float(time.time()),
        "command_template_file": str(ps1_path),
    }
    suggestions_path.write_text(json.dumps(to_jsonable(suggestions_payload), indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "template_path": ps1_path,
        "suggestions_path": suggestions_path,
        "overrides": overrides,
    }


def weight_block_signature(rep: AlgebraReport) -> List[Tuple[Tuple[str, ...], int]]:
    return sorted((tuple(block["weight"]), int(block["size"])) for block in rep.weight_blocks)


def analyze_algebra(c, invariant_level: str = "quick", compute_derivation_invariant: bool = False) -> AlgebraReport:
    n = len(c)
    center = center_subspace(c)
    derived = derived_subspace(c)
    if invariant_level == "full":
        lcs = lower_central_series_dims(c)
        ds = derived_series_dims(c)
    else:
        lcs = [n, len(derived)]
        ds = [len(derived)]

    # Fast adaptation: reorder basis so bracket outputs come first.
    # For target class, bracket-output span is a practical nilradical candidate.
    nil_indices = bracket_output_indices(c)
    nil_index_set = set(nil_indices)
    ext_indices = [i for i in range(n) if i not in nil_index_set]
    order = nil_indices + ext_indices
    nilradical_dim = len(nil_indices)
    extension_dim = n - nilradical_dim
    c_adapted = permute_structure_constants(c, order)
    c_adapted, ext_order, nil_order, weight_blocks = joint_weight_normalize(c_adapted, nilradical_dim, n)
    adapted_ctx = build_adapted_profile_context(c_adapted, nilradical_dim, n)
    nil_weight_labels: List[Tuple[str, ...]] = []
    for block in weight_blocks:
        nil_weight_labels.extend([tuple(block["weight"])] * int(block["size"]))
    action_ranks_ctx = [int(x) for x in _ctx_get_list(adapted_ctx, "action_ranks")]
    action_traces_ctx = list(_ctx_get_list(adapted_ctx, "action_traces"))
    if len(action_ranks_ctx) == extension_dim and len(action_traces_ctx) == extension_dim:
        extension_labels = [
            (str(sp.simplify(action_traces_ctx[i])), str(action_ranks_ctx[i]))
            for i in range(extension_dim)
        ]
    else:
        extension_labels = precompute_extension_action_stats(c_adapted, nilradical_dim, n)
    nil_kernel_labels = nil_kernel_block_labels(c_adapted, nilradical_dim)
    nil_v_labels = nil_vertex_color_labels(c_adapted, nilradical_dim, n)
    nil_edges = nil_colored_hyperedges(c_adapted, nilradical_dim, n)
    der_dim = derivation_dimension(c) if compute_derivation_invariant else None
    ext_w_multiset = extension_weight_multiset(c_adapted, nilradical_dim, n)

    return AlgebraReport(
        n=n,
        center_dim=len(center),
        derived_dim=len(derived),
        lower_central_dims=lcs,
        derived_series_dims=ds,
        nilradical_candidate_dim=nilradical_dim,
        extension_dim=extension_dim,
        adapted_structure_constants=c_adapted,
        n_basis_dim=nilradical_dim,
        extension_order_adapted_1based=ext_order,
        nil_order_adapted_1based=nil_order,
        weight_blocks=weight_blocks,
        nil_weight_labels=nil_weight_labels,
        extension_labels=extension_labels,
        nil_kernel_labels=nil_kernel_labels,
        nil_vertex_color_labels=nil_v_labels,
        nil_colored_hyperedges=nil_edges,
        derivation_dim=der_dim,
        extension_weight_multiset=ext_w_multiset,
        adapted_profile_context=adapted_ctx,
    )


def center_of_nilradical(ca, m: int) -> List[Vector]:
    z = [sp.Symbol(f"u{i+1}") for i in range(m)]
    eqs = []
    for j in range(m):
        for k in range(m):
            eqs.append(sum(z[i] * ca[i][j][k] for i in range(m)))
    A, _ = sp.linear_eq_to_matrix(eqs, z)
    return A.nullspace()


def bracket_tensor_profile(ca, m: int, ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    # Fast path for 2-step kernels already in an adapted basis:
    # if [nil,nil] lands entirely in the center, coordinates are read directly
    # from structure constants, avoiding repeated linear solves.
    ctx = ctx or build_adapted_profile_context(ca, m, len(ca))
    fast_2step = ctx.get("fast_2step")
    if isinstance(fast_2step, dict):
        return dict(fast_2step)
    raw = ctx.get("fast_2step_raw")
    if isinstance(raw, dict):
        zdim = int(raw.get("Z_dim", 0))
        vdim = int(raw.get("V_dim", 0))
        pair_count = int(raw.get("pair_count", 0))
        component_mats = raw.get("component_mats", [])
        all_coords_cols = raw.get("all_coords_cols", [])

        map_rank = 0
        if isinstance(all_coords_cols, list) and all_coords_cols:
            map_rank = sp.Matrix.hstack(*all_coords_cols).rank()

        component_ranks = []
        if isinstance(component_mats, list):
            for M in component_mats:
                if isinstance(M, sp.MatrixBase):
                    component_ranks.append(int(M.rank()))

        exact = {
            "Z_dim": zdim,
            "V_dim": vdim,
            "pair_count": pair_count,
            "map_rank": int(map_rank),
            "component_ranks": component_ranks,
            "profile_mode": "fast_2step",
        }
        ctx["fast_2step"] = exact
        return dict(exact)

    z_basis = center_of_nilradical(ca, m)
    zdim = len(z_basis)
    if zdim == 0:
        return {
            "Z_dim": 0,
            "V_dim": m,
            "pair_count": m * (m - 1) // 2,
            "map_rank": 0,
            "component_ranks": [],
        }

    Z = sp.Matrix.hstack(*z_basis)
    V_basis = complete_basis(z_basis, m)[zdim:]
    vdim = len(V_basis)

    component_mats = [sp.zeros(vdim, vdim) for _ in range(zdim)]
    all_coords_cols = []
    for i in range(vdim):
        for j in range(i + 1, vdim):
            vi = V_basis[i]
            vj = V_basis[j]
            bracket = sp.Matrix([sum(vi[a] * vj[b] * ca[a][b][k] for a in range(m) for b in range(m)) for k in range(m)])
            coords = Z.gauss_jordan_solve(bracket)[0]
            all_coords_cols.append(coords)
            for r in range(zdim):
                component_mats[r][i, j] = sp.simplify(coords[r])
                component_mats[r][j, i] = sp.simplify(-coords[r])

    map_rank = 0
    if all_coords_cols:
        map_rank = sp.Matrix.hstack(*all_coords_cols).rank()

    return {
        "Z_dim": zdim,
        "V_dim": vdim,
        "pair_count": vdim * (vdim - 1) // 2,
        "map_rank": int(map_rank),
        "component_ranks": [int(M.rank()) for M in component_mats],
        "profile_mode": "generic",
    }


def extension_action_profile(ca, m: int, n: int, level: str = "quick", ctx: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    r = n - m
    ctx = ctx or build_adapted_profile_context(ca, m, n)
    action_mats = [mat for mat in _ctx_get_list(ctx, "action_mats") if isinstance(mat, sp.MatrixBase)]
    action_ranks = [int(x) for x in _ctx_get_list(ctx, "action_ranks")]
    action_traces = list(_ctx_get_list(ctx, "action_traces"))

    if len(action_mats) != r:
        action_mats = [sp.Matrix([[ca[j][t_idx][i] for j in range(m)] for i in range(m)]) for t_idx in range(m, n)]
    if len(action_ranks) != r:
        action_ranks = [int(A.rank()) for A in action_mats]
    if len(action_traces) != r:
        action_traces = [sp.simplify(A.trace()) for A in action_mats]

    actions = []
    for pos, A in enumerate(action_mats):
        t_idx = m + pos
        item = {
            "t_index_adapted_1based": t_idx + 1,
            "rank": int(action_ranks[pos]),
            "trace": action_traces[pos],
        }
        if level == "full":
            item["det"] = sp.simplify(A.det())
            charpoly = sp.expand(A.charpoly().as_expr())
            item["charpoly"] = str(charpoly)
            try:
                jcells = A.jordan_cells()
                block_sizes = sorted([blk.rows for blk in jcells[1]], reverse=True)
            except Exception:
                block_sizes = []
            item["jordan_block_sizes"] = block_sizes
        actions.append(item)

    joint_plane = None
    if len(actions) == 2 and len(action_mats) >= 2:
        A1 = action_mats[0]
        A2 = action_mats[1]
        stacked = A1.col_join(A2)
        common_kernel_dim = int(m - stacked.rank())
        image_sum_dim = int(sp.Matrix.hstack(A1, A2).rank())
        comm = A1 * A2 - A2 * A1
        sample = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)]
        combo_sig = []
        for a, b in sample:
            M = a * A1 + b * A2
            combo_sig.append((a, b, int(M.rank()), str(sp.simplify(M.trace()))))
        joint_plane = {
            "common_kernel_dim": common_kernel_dim,
            "image_sum_dim": image_sum_dim,
            "commutator_rank": int(comm.rank()),
            "combo_rank_trace_signature": combo_sig,
        }

    return {
        "extension_dim": r,
        "level": level,
        "actions": actions,
        "joint_plane": joint_plane,
    }


def autn_oriented_signature(
    ca,
    m: int,
    n: int,
    weight_blocks: Sequence[Dict[str, object]],
    ctx: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    ctx = ctx or build_adapted_profile_context(ca, m, n)
    action_mats = [mat for mat in _ctx_get_list(ctx, "action_mats") if isinstance(mat, sp.MatrixBase)]
    if len(action_mats) != (n - m):
        action_mats = [sp.Matrix([[ca[j][t_idx][i] for j in range(m)] for i in range(m)]) for t_idx in range(m, n)]

    block_ranges: List[Tuple[int, int]] = []
    block_sig: List[Tuple[Tuple[str, ...], int]] = []
    offset = 0
    for blk in weight_blocks:
        size = int(blk.get("size", 0))
        weight = tuple(str(x) for x in blk.get("weight", []))
        block_ranges.append((offset, offset + size))
        block_sig.append((weight, size))
        offset += size

    if offset != m or not block_ranges:
        block_ranges = [(0, m)]
        block_sig = [(("all",), m)]

    def _block_extract(A: sp.MatrixBase, r0: int, r1: int, c0: int, c1: int) -> sp.MatrixBase:
        return A.extract(list(range(r0, r1)), list(range(c0, c1)))

    action_block_sigs = []
    for A in action_mats:
        diag_rank_trace = []
        offdiag_rank = []
        for bi, (r0, r1) in enumerate(block_ranges):
            B = _block_extract(A, r0, r1, r0, r1)
            diag_rank_trace.append((bi, int(B.rank()), str(sp.simplify(B.trace()))))
        for bi, (r0, r1) in enumerate(block_ranges):
            for bj, (c0, c1) in enumerate(block_ranges):
                if bi == bj:
                    continue
                B = _block_extract(A, r0, r1, c0, c1)
                rk = int(B.rank())
                if rk > 0:
                    offdiag_rank.append((bi, bj, rk))
        action_block_sigs.append(
            {
                "diag_rank_trace": diag_rank_trace,
                "offdiag_rank": sorted(offdiag_rank),
            }
        )

    joint_block_sig = None
    if len(action_mats) == 2:
        A1 = action_mats[0]
        A2 = action_mats[1]
        per_block = []
        for bi, (r0, r1) in enumerate(block_ranges):
            B1 = _block_extract(A1, r0, r1, r0, r1)
            B2 = _block_extract(A2, r0, r1, r0, r1)
            size = r1 - r0
            stacked = B1.col_join(B2)
            common_kernel_dim = int(size - stacked.rank())
            image_sum_dim = int(sp.Matrix.hstack(B1, B2).rank())
            comm = B1 * B2 - B2 * B1
            combo = []
            for a, b in [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1)]:
                M = a * B1 + b * B2
                combo.append((a, b, int(M.rank()), str(sp.simplify(M.trace()))))
            per_block.append(
                {
                    "block": bi,
                    "size": size,
                    "common_kernel_dim": common_kernel_dim,
                    "image_sum_dim": image_sum_dim,
                    "commutator_rank": int(comm.rank()),
                    "combo_rank_trace_signature": combo,
                }
            )
        joint_block_sig = per_block

    action_block_sigs_sorted = sorted(
        action_block_sigs,
        key=lambda x: (tuple(x.get("diag_rank_trace", [])), tuple(x.get("offdiag_rank", []))),
    )

    return {
        "block_signature": block_sig,
        "action_block_signatures": action_block_sigs_sorted,
        "joint_block_signature": joint_block_sig,
    }


def matrix_pencil_signature(ca, m: int, ctx: Optional[Dict[str, object]] = None, basis_cap: int = 3) -> Dict[str, object]:
    ctx = ctx or build_adapted_profile_context(ca, m, len(ca))
    raw = ctx.get("fast_2step_raw")
    if not isinstance(raw, dict):
        return {
            "available": False,
            "reason": "fast_2step_not_applicable",
            "samples": [],
            "sample_rank_signature": [],
        }
    component_mats = raw.get("component_mats", [])
    if not isinstance(component_mats, list) or not component_mats:
        return {
            "available": False,
            "reason": "no_component_mats",
            "samples": [],
            "sample_rank_signature": [],
        }

    zdim = len(component_mats)
    samples: List[Tuple[int, ...]] = []
    # Keep sampling bounded for stable pair latency.
    basis_cap_eff = max(1, min(zdim, int(basis_cap)))
    for i in range(basis_cap_eff):
        vec = [0] * zdim
        vec[i] = 1
        samples.append(tuple(vec))
    if zdim >= 2:
        samples.extend([
            tuple([1] * zdim),
            tuple([1 if i % 2 == 0 else -1 for i in range(zdim)]),
        ])

    rank_sig = []
    for coeffs in samples:
        M = sp.zeros(component_mats[0].rows, component_mats[0].cols)
        for a, comp in zip(coeffs, component_mats):
            if a:
                M += a * comp
        rank_sig.append((coeffs, int(M.rank())))

    return {
        "available": True,
        "reason": None,
        "samples": samples,
        "sample_rank_signature": rank_sig,
    }


def compare_reports(rep1: AlgebraReport, rep2: AlgebraReport, bt1: Dict[str, object], bt2: Dict[str, object], ext1: Dict[str, object], ext2: Dict[str, object]) -> Dict[str, object]:
    checks = {}
    checks["dim"] = rep1.n == rep2.n
    checks["center_dim"] = rep1.center_dim == rep2.center_dim
    checks["derived_dim"] = rep1.derived_dim == rep2.derived_dim
    if rep1.derived_series_dims and rep2.derived_series_dims:
        checks["derived_series_dims"] = rep1.derived_series_dims == rep2.derived_series_dims
    if rep1.lower_central_dims and rep2.lower_central_dims:
        checks["lower_central_dims"] = rep1.lower_central_dims == rep2.lower_central_dims
    checks["nilradical_candidate_dim"] = rep1.nilradical_candidate_dim == rep2.nilradical_candidate_dim
    checks["extension_dim"] = rep1.extension_dim == rep2.extension_dim
    checks["weight_blocks"] = weight_block_signature(rep1) == weight_block_signature(rep2)
    if rep1.derivation_dim is not None and rep2.derivation_dim is not None:
        checks["derivation_dim"] = rep1.derivation_dim == rep2.derivation_dim
    checks["extension_weight_multiset"] = rep1.extension_weight_multiset == rep2.extension_weight_multiset

    checks["bracket_tensor.Z_dim"] = bt1["Z_dim"] == bt2["Z_dim"]
    checks["bracket_tensor.map_rank"] = bt1["map_rank"] == bt2["map_rank"]
    checks["bracket_tensor.component_ranks"] = bt1["component_ranks"] == bt2["component_ranks"]

    if ext1.get("level") == "full" and ext2.get("level") == "full":
        ext_sig1 = [(a["rank"], str(a.get("charpoly", "")), a.get("jordan_block_sizes", [])) for a in ext1["actions"]]
        ext_sig2 = [(a["rank"], str(a.get("charpoly", "")), a.get("jordan_block_sizes", [])) for a in ext2["actions"]]
    else:
        ext_sig1 = [(a["rank"], str(a.get("trace", 0))) for a in ext1["actions"]]
        ext_sig2 = [(a["rank"], str(a.get("trace", 0))) for a in ext2["actions"]]
    checks["extension_action_signatures"] = sorted(ext_sig1) == sorted(ext_sig2)
    checks["extension_joint_plane_signature"] = ext1.get("joint_plane") == ext2.get("joint_plane")

    linear_pass = all(checks.values())
    return {
        "checks": checks,
        "linear_stage_pass": linear_pass,
    }


def base_invariant_checks(rep1: AlgebraReport, rep2: AlgebraReport) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    checks["dim"] = rep1.n == rep2.n
    checks["center_dim"] = rep1.center_dim == rep2.center_dim
    checks["derived_dim"] = rep1.derived_dim == rep2.derived_dim
    if rep1.derived_series_dims and rep2.derived_series_dims:
        checks["derived_series_dims"] = rep1.derived_series_dims == rep2.derived_series_dims
    if rep1.lower_central_dims and rep2.lower_central_dims:
        checks["lower_central_dims"] = rep1.lower_central_dims == rep2.lower_central_dims
    checks["nilradical_candidate_dim"] = rep1.nilradical_candidate_dim == rep2.nilradical_candidate_dim
    checks["extension_dim"] = rep1.extension_dim == rep2.extension_dim
    checks["weight_blocks"] = weight_block_signature(rep1) == weight_block_signature(rep2)
    checks["extension_weight_multiset"] = rep1.extension_weight_multiset == rep2.extension_weight_multiset
    if rep1.derivation_dim is not None and rep2.derivation_dim is not None:
        checks["derivation_dim"] = rep1.derivation_dim == rep2.derivation_dim
    return checks


def generate_block_symbolic_system(
    ca1,
    ca2,
    m: int,
    n: int,
    max_equations: int = 500,
    max_pairs: int = 120,
    nil_weight_labels1: Sequence[Tuple[str, ...]] | None = None,
    nil_weight_labels2: Sequence[Tuple[str, ...]] | None = None,
    extension_labels1: Sequence[Tuple[str, ...]] | None = None,
    extension_labels2: Sequence[Tuple[str, ...]] | None = None,
    nil_kernel_labels1: Sequence[Tuple[str, ...]] | None = None,
    nil_kernel_labels2: Sequence[Tuple[str, ...]] | None = None,
    same_kernel_mode: bool = False,
) -> Dict[str, object]:
    r = n - m

    def xn_entry(i: int, j: int):
        if nil_weight_labels1 is not None and nil_weight_labels2 is not None:
            if tuple(nil_weight_labels2[i]) != tuple(nil_weight_labels1[j]):
                return sp.Integer(0)
        if same_kernel_mode and nil_kernel_labels1 is not None and nil_kernel_labels2 is not None:
            if tuple(nil_kernel_labels2[i]) != tuple(nil_kernel_labels1[j]):
                return sp.Integer(0)
        return sp.Symbol(f"xN_{i+1}_{j+1}")

    def xt_entry(i: int, j: int):
        if extension_labels1 is not None and extension_labels2 is not None:
            if tuple(extension_labels2[i]) != tuple(extension_labels1[j]):
                return sp.Integer(0)
        return sp.Symbol(f"xT_{i+1}_{j+1}")

    xn = sp.Matrix(m, m, xn_entry)
    xt = sp.Matrix(r, r, xt_entry)
    y = sp.Matrix(m, r, lambda i, j: sp.Symbol(f"y_{i+1}_{j+1}"))

    X = sp.zeros(n)
    X[:m, :m] = xn
    X[:m, m:n] = y
    X[m:n, m:n] = xt

    equations = []

    def bracket_c(ca, u: Vector, v: Vector) -> Vector:
        out = sp.Matrix([0] * n)
        for i in range(n):
            if u[i] == 0:
                continue
            for j in range(n):
                if v[j] == 0:
                    continue
                coeff = u[i] * v[j]
                if coeff == 0:
                    continue
                for k in range(n):
                    if ca[i][j][k] != 0:
                        out[k] += coeff * ca[i][j][k]
        return out

    e = [sp.Matrix([1 if t == i else 0 for t in range(n)]) for i in range(n)]

    pair_counter = 0
    for i in range(n):
        Xi = X * e[i]
        for j in range(i + 1, n):
            if pair_counter >= max_pairs:
                break
            Xj = X * e[j]
            lhs = X * sp.Matrix([ca1[i][j][k] for k in range(n)])
            rhs = bracket_c(ca2, Xi, Xj)
            diff = lhs - rhs
            for k in range(n):
                if diff[k] != 0:
                    equations.append(diff[k])
            pair_counter += 1
        if pair_counter >= max_pairs:
            break

    equations = [eq for eq in equations if eq != 0]

    def dedupe_equations(eqs):
        seen = set()
        out = []
        for eq in eqs:
            e = sp.expand(eq)
            if e == 0:
                continue
            key = str(e)
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def singleton_zero_symbol(eq):
        e = sp.expand(eq)
        fsyms = list(e.free_symbols)
        if len(fsyms) != 1:
            return None
        s = fsyms[0]
        try:
            p = sp.Poly(e, s)
        except Exception:
            return None
        # a*s = 0 pattern (constant term is zero)
        if p.degree() == 1 and p.nth(0) == 0:
            return s
        return None

    equations = dedupe_equations(equations)

    xn_symbol_count = sum(1 for i in range(m) for j in range(m) if isinstance(xn[i, j], sp.Symbol))
    xt_symbol_count = sum(1 for i in range(r) for j in range(r) if isinstance(xt[i, j], sp.Symbol))
    vars_list = [xn[i, j] for i in range(m) for j in range(m) if isinstance(xn[i, j], sp.Symbol)]
    vars_list += [xt[i, j] for i in range(r) for j in range(r) if isinstance(xt[i, j], sp.Symbol)]
    vars_list += [y[i, j] for i in range(m) for j in range(r)]
    vars_set = set(vars_list)

    forced_zero = set()
    changed = True
    while changed:
        changed = False
        new_forced = set()
        for eq in equations:
            s = singleton_zero_symbol(eq)
            if s is not None and s in vars_set and s not in forced_zero:
                new_forced.add(s)
        if new_forced:
            forced_zero |= new_forced
            sub_map = {s: 0 for s in new_forced}
            equations = [sp.expand(eq.subs(sub_map)) for eq in equations]
            equations = dedupe_equations(equations)
            changed = True

    active_symbols = set()
    for eq in equations:
        active_symbols |= (eq.free_symbols & vars_set)
    compressed_vars = [v for v in vars_list if v in active_symbols]

    def count_block_vars(block, symbols):
        block_syms = set(block)
        return sum(1 for s in symbols if s in block_syms)

    unknown_before = len(vars_list)
    unknown_after = len(compressed_vars)
    equations_before = pair_counter * n
    equations_after = len(equations)
    payload = {
        "unknown_blocks": {
            "Xn_shape": [m, m],
            "Xt_shape": [r, r],
            "Y_shape": [m, r],
            "unknown_count": unknown_after,
            "unknown_count_before": unknown_before,
            "active_unknowns_by_block": {
                "Xn": count_block_vars(xn, compressed_vars),
                "Xt": count_block_vars(xt, compressed_vars),
                "Y": count_block_vars(y, compressed_vars),
            },
            "forced_zero_count": len(forced_zero),
            "blocked_by_weight_count": (m * m - xn_symbol_count) + (r * r - xt_symbol_count),
        },
        "generation_limits": {
            "max_pairs": max_pairs,
            "max_equations_preview": max_equations,
            "pairs_used": pair_counter,
        },
        "equation_count_total": equations_after,
        "equation_count_before": equations_before,
        "compression": {
            "dedup_and_substitution_applied": True,
            "forced_zero_symbols_preview": [str(s) for s in sorted(forced_zero, key=str)[: max_equations // 2]],
        },
        "equations_preview": [str(eq) for eq in equations[:max_equations]],
        "det_constraints": {
            "det_Xn_nonzero": "det(Xn) != 0",
            "det_Xt_nonzero": "det(Xt) != 0",
        },
        "_raw": {
            "equations": equations,
            "variables": compressed_vars,
        },
    }
    return payload


def _modp_inv(a: int, p: int) -> int:
    return pow(a % p, p - 2, p)


def modp_rank(mat: List[List[int]], p: int) -> int:
    if not mat:
        return 0
    A = [row[:] for row in mat]
    nrows = len(A)
    ncols = len(A[0])
    r = 0
    c = 0
    while r < nrows and c < ncols:
        pivot = None
        for i in range(r, nrows):
            if A[i][c] % p != 0:
                pivot = i
                break
        if pivot is None:
            c += 1
            continue
        A[r], A[pivot] = A[pivot], A[r]
        inv = _modp_inv(A[r][c], p)
        A[r] = [(v * inv) % p for v in A[r]]
        for i in range(nrows):
            if i == r:
                continue
            factor = A[i][c] % p
            if factor:
                A[i] = [(A[i][j] - factor * A[r][j]) % p for j in range(ncols)]
        r += 1
        c += 1
    return r


def extract_linear_system(equations: Sequence[sp.Expr], variables: Sequence[sp.Symbol]):
    var_index = {v: i for i, v in enumerate(variables)}
    A = []
    b = []
    for eq in equations:
        try:
            poly = sp.Poly(eq, *variables)
        except Exception:
            continue
        if poly.total_degree() > 1:
            continue
        row = [0] * len(variables)
        const = sp.Integer(0)
        ok = True
        for mon, coeff in poly.terms():
            deg = sum(mon)
            if deg == 0:
                const = coeff
            elif deg == 1:
                vidx = None
                for i, e in enumerate(mon):
                    if e == 1:
                        vidx = i
                        break
                if vidx is None:
                    ok = False
                    break
                row[vidx] = coeff
            else:
                ok = False
                break
        if ok:
            A.append(row)
            b.append(-const)
    return A, b


def modp_fingerprint(equations: Sequence[sp.Expr], variables: Sequence[sp.Symbol], primes: Sequence[int]) -> Dict[str, object]:
    A, b = extract_linear_system(equations, variables)
    fp = {
        "primes": list(primes),
        "linear_equation_count": len(A),
        "variable_count": len(variables),
        "by_prime": [],
    }
    for p in primes:
        if A:
            A_mod = [[int(sp.Integer(v) % p) for v in row] for row in A]
            b_mod = [int(sp.Integer(v) % p) for v in b]
            aug = [row + [b_mod[i]] for i, row in enumerate(A_mod)]
            rank_A = modp_rank(A_mod, p)
            rank_aug = modp_rank(aug, p)
            consistent = rank_A == rank_aug
        else:
            rank_A = 0
            rank_aug = 0
            consistent = True

        # Nonlinear fingerprint: rank of Jacobian at a deterministic random point mod p.
        sample_vars = list(variables[: min(48, len(variables))])
        sample_eqs = list(equations[: min(96, len(equations))])
        rng = random.Random(p)
        point = {v: rng.randrange(1, p) for v in variables}
        jac_rows = []
        for eq in sample_eqs:
            row = []
            for v in sample_vars:
                val = sp.diff(eq, v).subs(point)
                row.append(int(sp.Integer(val) % p))
            jac_rows.append(row)
        jac_rank = modp_rank(jac_rows, p) if jac_rows and sample_vars else 0
        nullity = len(variables) - rank_A
        fp["by_prime"].append(
            {
                "p": p,
                "rank_A": rank_A,
                "rank_aug": rank_aug,
                "consistent": bool(consistent),
                "nullity": nullity,
                "jacobian_rank_random_point": jac_rank,
                "jacobian_rows": len(jac_rows),
                "jacobian_cols": len(sample_vars),
            }
        )
    return fp


def choose_numeric_subsystem(equations: Sequence[sp.Expr], variables: Sequence[sp.Symbol], max_vars: int, max_eqs: int):
    freq = {v: 0 for v in variables}
    for eq in equations:
        for s in eq.free_symbols:
            if s in freq:
                freq[s] += 1
    selected_vars = sorted(variables, key=lambda v: freq[v], reverse=True)[:max_vars]
    selected_set = set(selected_vars)
    selected_eqs = []
    for eq in equations:
        fs = eq.free_symbols
        if not fs:
            continue
        if fs & selected_set:
            selected_eqs.append(eq)
            if len(selected_eqs) >= max_eqs:
                break
    # Freeze unselected variables at a sparse identity-like seed.
    frozen = {}
    for v in variables:
        if v in selected_set:
            continue
        name = str(v)
        seed = 0
        if name.startswith("xN_") or name.startswith("xT_"):
            parts = name.split("_")
            if len(parts) == 3 and parts[1] == parts[2]:
                seed = 1
        frozen[v] = seed
    selected_eqs = [sp.expand(eq.subs(frozen)) for eq in selected_eqs]
    selected_eqs = [eq for eq in selected_eqs if eq != 0]

    # Keep only selected variables that actually occur.
    used = set()
    for eq in selected_eqs:
        used |= eq.free_symbols
    selected_vars = [v for v in selected_vars if v in used]
    if len(selected_eqs) < len(selected_vars):
        selected_vars = selected_vars[: len(selected_eqs)]
        selected_set = set(selected_vars)
        selected_eqs = [sp.expand(eq.subs({v: frozen.get(v, 0) for v in used if v not in selected_set})) for eq in selected_eqs]
    return selected_vars, selected_eqs[:max_eqs], frozen


def numeric_candidate_search(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    max_vars: int = 18,
    max_eqs: int = 18,
    restarts: int = 8,
) -> Dict[str, object]:
    seed_map = {}
    for v in variables:
        name = str(v)
        seed = 0
        if name.startswith("xN_") or name.startswith("xT_"):
            parts = name.split("_")
            if len(parts) == 3 and parts[1] == parts[2]:
                seed = 1
        seed_map[v] = sp.Integer(seed)

    seed_residuals = [sp.simplify(eq.subs(seed_map)) for eq in equations]
    seed_nonzero = [value for value in seed_residuals if value != 0]
    if not seed_nonzero:
        return {
            "selected_variable_count": 0,
            "selected_equation_count": 0,
            "success": True,
            "method": "structured identity seed",
            "max_residual_reduced_numeric": 0.0,
            "candidate_preview": {str(v): str(seed_map[v]) for v in list(variables)[:12]},
            "_raw": {
                "numeric_map": seed_map,
                "rational_map": seed_map,
                "selected_vars": [],
                "selected_eqs": [],
            },
        }

    violated_equations = [
        equations[i]
        for i, value in enumerate(seed_residuals)
        if value != 0
    ]
    search_equations = violated_equations + [
        eq for i, eq in enumerate(equations) if seed_residuals[i] == 0
    ]
    vars_sel, eqs_sel, frozen = choose_numeric_subsystem(
        search_equations,
        variables,
        max_vars=max_vars,
        max_eqs=max_eqs,
    )
    result = {
        "selected_variable_count": len(vars_sel),
        "selected_equation_count": len(eqs_sel),
        "success": False,
        "method": "sympy.nsolve",
            "frozen_variable_count": len(frozen),
            "structured_seed_nonzero_residual_count": len(seed_nonzero),
    }
    if not vars_sel or not eqs_sel:
        result["note"] = "no suitable reduced subsystem for numeric solve"
        return result

    best = None
    best_res = None
    for _ in range(restarts):
        guess = [sp.Float(random.uniform(-0.5, 0.5)) for _ in vars_sel]
        try:
            sol = sp.nsolve(eqs_sel, vars_sel, guess, tol=1e-12, maxsteps=80, prec=40)
            if isinstance(sol, (list, tuple)):
                vals = [sp.N(v, 30) for v in sol]
            else:
                vals = [sp.N(sol[i], 30) for i in range(sol.rows)]
            res = max(abs(complex(sp.N(eq.subs({vars_sel[i]: vals[i] for i in range(len(vars_sel))}, 30)))) for eq in eqs_sel)
            if best_res is None or res < best_res:
                best = vals
                best_res = res
            if res < 1e-8:
                break
        except Exception:
            continue

    if best is None:
        result["note"] = "numeric solve failed on reduced subsystem"
        return result

    num_map = {**frozen, **{vars_sel[i]: best[i] for i in range(len(vars_sel))}}
    rat_map = {v: sp.nsimplify(num_map[v], rational=True) for v in vars_sel}
    rat_map.update({v: sp.Integer(value) for v, value in frozen.items()})
    result.update(
        {
            "success": True,
            "max_residual_reduced_numeric": float(best_res),
            "candidate_preview": {str(v): str(rat_map[v]) for v in vars_sel[: min(12, len(vars_sel))]},
            "_raw": {
                "numeric_map": num_map,
                "rational_map": rat_map,
                "selected_vars": vars_sel,
                "selected_eqs": eqs_sel,
            },
        }
    )
    return result


def adaptive_numeric_candidate_search(
    equations: Sequence[sp.Expr],
    variables: Sequence[sp.Symbol],
    base_max_vars: int = 18,
    base_max_eqs: int = 18,
    base_restarts: int = 8,
) -> Dict[str, object]:
    var_cap = len(variables)
    eq_cap = len(equations)
    configs = [
        (min(var_cap, max(4, base_max_vars)), min(eq_cap, max(4, base_max_eqs)), max(2, base_restarts)),
        (
            min(var_cap, max(base_max_vars + 12, int(base_max_vars * 1.3))),
            min(eq_cap, max(base_max_eqs + 30, int(base_max_eqs * 1.5))),
            max(4, base_restarts * 2),
        ),
        (
            min(var_cap, max(base_max_vars + 24, int(base_max_vars * 1.8))),
            min(eq_cap, max(base_max_eqs + 80, int(base_max_eqs * 2.0))),
            max(6, base_restarts * 3),
        ),
    ]

    # Remove duplicates while preserving order.
    seen = set()
    uniq_configs = []
    for cfg in configs:
        if cfg in seen:
            continue
        seen.add(cfg)
        uniq_configs.append(cfg)

    attempts = []
    for max_vars, max_eqs, restarts in uniq_configs:
        print(
            f"[adaptive] try max_vars={max_vars}, max_eqs={max_eqs}, restarts={restarts}",
            flush=True,
        )
        res = numeric_candidate_search(
            equations,
            variables,
            max_vars=max_vars,
            max_eqs=max_eqs,
            restarts=restarts,
        )
        attempt = {
            "max_vars": max_vars,
            "max_eqs": max_eqs,
            "restarts": restarts,
            "selected_variable_count": int(res.get("selected_variable_count", 0)),
            "selected_equation_count": int(res.get("selected_equation_count", 0)),
            "success": bool(res.get("success", False)),
            "note": str(res.get("note", "")),
        }
        if "max_residual_reduced_numeric" in res:
            attempt["max_residual_reduced_numeric"] = float(res["max_residual_reduced_numeric"])
        attempts.append(attempt)
        if res.get("success", False):
            print("[adaptive] numeric candidate found", flush=True)
            out = dict(res)
            out["adaptive_attempts"] = attempts
            out["adaptive_mode"] = True
            return out

    print("[adaptive] all tiers failed", flush=True)
    out = dict(attempts[-1]) if attempts else {
        "selected_variable_count": 0,
        "selected_equation_count": 0,
        "success": False,
    }
    out.update(
        {
            "success": False,
            "method": "sympy.nsolve",
            "note": "adaptive numeric solve failed on all tiers",
            "adaptive_attempts": attempts,
            "adaptive_mode": True,
        }
    )
    return out


def exact_back_substitute_verify(equations: Sequence[sp.Expr], candidate_rat_map: Dict[sp.Symbol, sp.Expr], variables: Sequence[sp.Symbol]) -> Dict[str, object]:
    # Unassigned variables are set to 0 in this fast verification stage.
    full_map = {v: candidate_rat_map.get(v, sp.Integer(0)) for v in variables}
    nonzero = 0
    preview = []
    for eq in equations:
        val = sp.simplify(eq.subs(full_map))
        if val != 0:
            nonzero += 1
            if len(preview) < 12:
                preview.append(str(val))
    return {
        "scope": "compressed candidate equations generated in this run",
        "assigned_variable_count": len(candidate_rat_map),
        "verification_equation_count": len(equations),
        "exact_zero_count": len(equations) - nonzero,
        "exact_nonzero_count": nonzero,
        "residual_preview": preview,
    }


def exact_sparse_monomial_search(c1, c2, nil_dim: int) -> Dict[str, object]:
    """Search an exact signed-permutation isomorphism for sparse graph-type 2-step kernels."""
    n = len(c1)
    if len(c2) != n:
        return {"applicable": False, "success": False, "note": "dimension mismatch"}
    ext = list(range(nil_dim, n))

    def kernel_center_indices(c):
        return [
            i for i in range(nil_dim)
            if all(c[i][j][k] == 0 for j in range(nil_dim) for k in range(nil_dim))
        ]

    z1 = kernel_center_indices(c1)
    z2 = kernel_center_indices(c2)
    z1_set, z2_set = set(z1), set(z2)
    v1 = [i for i in range(nil_dim) if i not in z1_set]
    v2 = [i for i in range(nil_dim) if i not in z2_set]
    if len(v1) != len(v2) or len(z1) != len(z2) or len(v1) > 9:
        return {
            "applicable": False,
            "success": False,
            "note": "requires equal graph-type kernels with at most 9 noncentral generators",
        }

    def action_color(c, i):
        return tuple(sp.simplify(c[i][t][i]) for t in ext)

    def local_degree(c, idx):
        d = 0
        for j in range(nil_dim):
            if j == idx:
                continue
            if any(c[idx][j][k] != 0 for k in range(nil_dim)):
                d += 1
        return d

    def sparse_edges(c, vertices, centers):
        center_set = set(centers)
        result = {}
        for a_pos, i in enumerate(vertices):
            for b_pos in range(a_pos + 1, len(vertices)):
                j = vertices[b_pos]
                nz = [(k, sp.simplify(c[i][j][k])) for k in range(n) if c[i][j][k] != 0]
                if not nz:
                    continue
                if len(nz) != 1 or nz[0][0] not in center_set:
                    return None
                result[(a_pos, b_pos)] = (nz[0][0], nz[0][1])
        return result

    e1 = sparse_edges(c1, v1, z1)
    e2 = sparse_edges(c2, v2, z2)
    if e1 is None or e2 is None or len(e1) != len(e2) or len(e1) != len(z1):
        return {
            "applicable": False,
            "success": False,
            "note": "kernel is not in the supported one-edge/one-center sparse form",
        }

    vertex_colors1 = [(action_color(c1, i), local_degree(c1, i)) for i in v1]
    vertex_colors2 = [(action_color(c2, i), local_degree(c2, i)) for i in v2]

    # Build candidate lists by color class (hard pruning from linear/weight/action info).
    color_to_targets: Dict[Tuple[Tuple[sp.Expr, ...], int], List[int]] = {}
    for idx2, col2 in enumerate(vertex_colors2):
        color_to_targets.setdefault(col2, []).append(idx2)
    candidate_lists = []
    for idx1, col1 in enumerate(vertex_colors1):
        cands = color_to_targets.get(col1, [])
        if not cands:
            return {
                "applicable": True,
                "success": False,
                "method": "exact sparse monomial search",
                "permutations_tested": 0,
                "full_permutation_count": int(sp.factorial(len(v1))),
                "color_class_permutation_upper_bound": 0,
                "note": "no feasible mapping after color-class pruning",
            }
        candidate_lists.append((idx1, list(cands), int(col1[1])))

    class_sizes: Dict[Tuple[Tuple[sp.Expr, ...], int], int] = {}
    for col in vertex_colors1:
        class_sizes[col] = class_sizes.get(col, 0) + 1
    color_upper = 1
    for sz in class_sizes.values():
        color_upper *= int(sp.factorial(sz))
    full_perm = int(sp.factorial(len(v1)))

    # IR heuristic: smallest candidate set first, and for ties branch on higher local degree first.
    candidate_lists.sort(key=lambda x: (len(x[1]), -x[2]))
    order = [x[0] for x in candidate_lists]
    ordered_cands = [x[1] for x in candidate_lists]
    pos_in_order = {v: i for i, v in enumerate(order)}

    # Symmetry pruning proxy: identical color classes on source side.
    source_class: Dict[Tuple[Tuple[sp.Expr, ...], int], List[int]] = {}
    for i, col in enumerate(vertex_colors1):
        source_class.setdefault(col, []).append(i)

    permutations_tested = 0
    failed_partial_states: set = set()

    def partial_state_key(assign: Dict[int, int]) -> Tuple[Tuple[int, int], ...]:
        # Cache failure states by sorted partial mapping on source vertex positions.
        return tuple(sorted((int(k), int(v)) for k, v in assign.items()))

    def backtrack(depth: int, used: set, assign: Dict[int, int]):
        nonlocal permutations_tested
        state_key = partial_state_key(assign)
        if state_key in failed_partial_states:
            return None
        if depth == len(order):
            permutations_tested += 1
            perm = [assign[i] for i in range(len(v1))]
            return perm

        src = order[depth]
        col = vertex_colors1[src]
        tried_targets = set()
        for tgt in ordered_cands[depth]:
            if tgt in used:
                continue
            # Orbit-style pruning: for same source color class at this depth, avoid repeating same target pattern.
            if tgt in tried_targets:
                continue

            # Early edge-consistency check against already assigned neighbors.
            ok = True
            for prev_src, prev_tgt in assign.items():
                a, b = sorted((src, prev_src))
                has1 = (a, b) in e1
                aa, bb = sorted((tgt, prev_tgt))
                has2 = (aa, bb) in e2
                if has1 != has2:
                    ok = False
                    break
                if has1 and has2:
                    z1_idx, _ = e1[(a, b)]
                    z2_idx, _ = e2[(aa, bb)]
                    if action_color(c1, z1_idx) != action_color(c2, z2_idx):
                        ok = False
                        break
            if not ok:
                continue

            assign[src] = tgt
            used.add(tgt)
            perm = backtrack(depth + 1, used, assign)
            if perm is not None:
                return perm
            used.remove(tgt)
            del assign[src]
            tried_targets.add(tgt)
        failed_partial_states.add(state_key)
        return None

    perm = backtrack(0, set(), {})
    if perm is None:
        return {
            "applicable": True,
            "success": False,
            "method": "exact sparse monomial search",
            "permutations_tested": permutations_tested,
            "full_permutation_count": full_perm,
            "color_class_permutation_upper_bound": color_upper,
            "note": "no signed-permutation candidate found after IR pruning; this does not rule out a general linear isomorphism",
            "failed_partial_state_count": len(failed_partial_states),
        }

    # Build center map and exact verification for the discovered permutation.
    center_map = {}
    used_targets = set()
    valid = True
    for (a, b), (source_z, source_coeff) in e1.items():
        pa, pb = perm[a], perm[b]
        key = (min(pa, pb), max(pa, pb))
        if key not in e2:
            valid = False
            break
        target_z, target_coeff = e2[key]
        if action_color(c1, source_z) != action_color(c2, target_z):
            valid = False
            break
        orientation = 1 if pa < pb else -1
        target_scale = sp.simplify(source_coeff / (orientation * target_coeff))
        center_map[source_z] = (target_z, target_scale)
        used_targets.add(target_z)
    if not valid or len(center_map) != len(z1) or len(used_targets) != len(z2):
        return {
            "applicable": True,
            "success": False,
            "method": "exact sparse monomial search",
            "permutations_tested": permutations_tested,
            "full_permutation_count": full_perm,
            "color_class_permutation_upper_bound": color_upper,
            "note": "candidate failed center-map compatibility",
            "failed_partial_state_count": len(failed_partial_states),
        }

    target = list(range(n))
    scale = [sp.Integer(1)] * n
    for i, source_i in enumerate(v1):
        target[source_i] = v2[perm[i]]
    for source_z, (target_z, target_scale) in center_map.items():
        target[source_z] = target_z
        scale[source_z] = target_scale

    bad_count = 0
    for i in range(n):
        for j in range(n):
            lhs = [sp.Integer(0)] * n
            rhs = [sp.Integer(0)] * n
            for k in range(n):
                if c1[i][j][k] != 0:
                    lhs[target[k]] += c1[i][j][k] * scale[k]
                if c2[target[i]][target[j]][k] != 0:
                    rhs[k] += scale[i] * scale[j] * c2[target[i]][target[j]][k]
            if lhs != rhs:
                bad_count += 1
    if bad_count == 0:
        return {
            "applicable": True,
            "success": True,
            "method": "exact sparse monomial search",
            "permutations_tested": permutations_tested,
            "full_permutation_count": full_perm,
            "color_class_permutation_upper_bound": color_upper,
            "exact_bracket_residual_count": 0,
            "basis_map_1based": [
                {"source": i + 1, "target": target[i] + 1, "scale": scale[i]}
                for i in range(n)
            ],
            "failed_partial_state_count": len(failed_partial_states),
        }

    return {
        "applicable": True,
        "success": False,
        "method": "exact sparse monomial search",
        "permutations_tested": permutations_tested,
        "full_permutation_count": full_perm,
        "color_class_permutation_upper_bound": color_upper,
        "note": "candidate failed exact global bracket verification",
        "failed_partial_state_count": len(failed_partial_states),
    }


def report_to_dict(rep: AlgebraReport) -> Dict[str, object]:
    return {
        "n": rep.n,
        "center_dim": rep.center_dim,
        "derived_dim": rep.derived_dim,
        "derived_series_dims": rep.derived_series_dims,
        "lower_central_dims": rep.lower_central_dims,
        "nilradical_candidate_dim": rep.nilradical_candidate_dim,
        "extension_dim": rep.extension_dim,
        "extension_order_adapted_1based": rep.extension_order_adapted_1based,
        "nil_order_adapted_1based": rep.nil_order_adapted_1based,
        "weight_block_signature": weight_block_signature(rep),
        "weight_blocks": rep.weight_blocks,
        "nil_kernel_labels_preview": [list(x) for x in rep.nil_kernel_labels[: min(12, len(rep.nil_kernel_labels))]],
        "derivation_dim": rep.derivation_dim,
        "extension_weight_multiset_preview": [list(x) for x in rep.extension_weight_multiset[: min(12, len(rep.extension_weight_multiset))]],
    }


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (sp.Integer, int)):
        return int(obj)
    if isinstance(obj, (sp.Rational,)):
        if obj.q == 1:
            return int(obj.p)
        return str(obj)
    if isinstance(obj, (sp.Float, float)):
        return float(obj)
    if isinstance(obj, sp.Basic):
        return str(obj)
    return obj


def _from_jsonable_sympy(obj):
    if isinstance(obj, list):
        return [_from_jsonable_sympy(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _from_jsonable_sympy(v) for k, v in obj.items()}
    if isinstance(obj, int):
        return sp.Integer(obj)
    if isinstance(obj, float):
        return sp.Rational(str(obj))
    if isinstance(obj, str):
        try:
            return sp.sympify(obj)
        except Exception:
            return obj
    return obj


def structure_constants_hash(c) -> str:
    n = len(c)
    triples = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v = c[i][j][k]
                if v != 0:
                    triples.append((i, j, k, str(v)))
    payload = json.dumps([n, triples], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def algebra_cache_key(
    c,
    n: int,
    invariant_level: str,
    extension_profile_level: str,
    compute_derivation_invariant: bool,
) -> str:
    h = structure_constants_hash(c)
    return "|".join(
        [
            f"n={n}",
            f"h={h}",
            f"inv={invariant_level}",
            f"ext={extension_profile_level}",
            f"der={1 if compute_derivation_invariant else 0}",
        ]
    )


def serialize_algebra_report(rep: AlgebraReport) -> Dict[str, object]:
    return to_jsonable(rep.__dict__)


def deserialize_algebra_report(payload: Dict[str, object]) -> Optional[AlgebraReport]:
    try:
        ca = _from_jsonable_sympy(payload.get("adapted_structure_constants", []))
        nil_weight_labels = [tuple(str(x) for x in row) for row in payload.get("nil_weight_labels", [])]
        extension_labels = [tuple(str(x) for x in row) for row in payload.get("extension_labels", [])]
        nil_kernel_labels = [tuple(str(x) for x in row) for row in payload.get("nil_kernel_labels", [])]
        nil_vertex_labels = [tuple(str(x) for x in row) for row in payload.get("nil_vertex_color_labels", [])]
        nil_edges = []
        for item in payload.get("nil_colored_hyperedges", []):
            if isinstance(item, list) and len(item) == 3:
                nil_edges.append((int(item[0]), int(item[1]), tuple(str(x) for x in item[2])))
        ext_w = [tuple(str(x) for x in row) for row in payload.get("extension_weight_multiset", [])]

        return AlgebraReport(
            n=int(payload.get("n", 0)),
            center_dim=int(payload.get("center_dim", 0)),
            derived_dim=int(payload.get("derived_dim", 0)),
            lower_central_dims=[int(x) for x in payload.get("lower_central_dims", [])],
            derived_series_dims=[int(x) for x in payload.get("derived_series_dims", [])],
            nilradical_candidate_dim=int(payload.get("nilradical_candidate_dim", 0)),
            extension_dim=int(payload.get("extension_dim", 0)),
            adapted_structure_constants=ca,
            n_basis_dim=int(payload.get("n_basis_dim", 0)),
            extension_order_adapted_1based=[int(x) for x in payload.get("extension_order_adapted_1based", [])],
            nil_order_adapted_1based=[int(x) for x in payload.get("nil_order_adapted_1based", [])],
            weight_blocks=payload.get("weight_blocks", []),
            nil_weight_labels=nil_weight_labels,
            extension_labels=extension_labels,
            nil_kernel_labels=nil_kernel_labels,
            nil_vertex_color_labels=nil_vertex_labels,
            nil_colored_hyperedges=nil_edges,
            derivation_dim=(int(payload["derivation_dim"]) if payload.get("derivation_dim") is not None else None),
            extension_weight_multiset=ext_w,
        )
    except Exception:
        return None


def _batch_process_worker(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    try:
        module_path = Path(str(payload["path"]))
        n = int(payload["n"])
        invariant_level = str(payload["invariant_level"])
        extension_profile_level = str(payload["extension_profile_level"])
        same_kernel_mode = bool(payload["same_kernel_mode"])
        derivation_mode = str(payload["derivation_invariant"])
        autn_sig_enabled = bool(payload.get("autn_sig_enabled", True))

        module = load_module(module_path)
        c = module.build_structure_constants(n)
        if derivation_mode == "on":
            use_derivation_invariant = True
        elif derivation_mode == "off":
            use_derivation_invariant = False
        else:
            use_derivation_invariant = not same_kernel_mode

        rep = analyze_algebra(
            c,
            invariant_level=invariant_level,
            compute_derivation_invariant=use_derivation_invariant,
        )
        ctx = build_adapted_profile_context(rep.adapted_structure_constants, rep.n_basis_dim, rep.n)
        bt = bracket_tensor_profile(rep.adapted_structure_constants, rep.n_basis_dim, ctx=ctx)
        mg = matrix_group_signature(rep.adapted_structure_constants, rep.n_basis_dim, ctx=ctx)
        ext = extension_action_profile(
            rep.adapted_structure_constants,
            rep.n_basis_dim,
            rep.n,
            level=extension_profile_level,
            ctx=ctx,
        )
        autn = autn_oriented_signature(
            rep.adapted_structure_constants,
            rep.n_basis_dim,
            rep.n,
            rep.weight_blocks,
            ctx=ctx,
        ) if autn_sig_enabled else None

        key = algebra_cache_key(c, n, invariant_level, extension_profile_level, use_derivation_invariant)
        return {
            "path": str(module_path),
            "n": n,
            "key": key,
            "rep": serialize_algebra_report(rep),
            "bt": to_jsonable(bt),
            "mg": to_jsonable(mg),
            "ext": to_jsonable(ext),
            "autn": to_jsonable(autn) if autn_sig_enabled else None,
        }
    except Exception:
        return None


def load_artifact_cache(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    if not isinstance(data, dict):
        return {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    if int(data.get("version", 0)) != CACHE_SCHEMA_VERSION or str(data.get("algo_tag", "")) != CACHE_ALGO_TAG:
        return {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    return {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": entries}


def save_artifact_cache(path: Path, cache: Dict[str, object]) -> None:
    cache["version"] = CACHE_SCHEMA_VERSION
    cache["algo_tag"] = CACHE_ALGO_TAG
    path.write_text(json.dumps(to_jsonable(cache), ensure_ascii=False, indent=2), encoding="utf-8")


def run_pair_pipeline(args_dict: Dict[str, object]) -> Dict[str, object]:
    t0_pipeline = time.perf_counter()
    alg1 = str(args_dict["alg1"])
    alg2 = str(args_dict["alg2"])
    n = int(args_dict.get("n", 23))

    mod1 = load_module(Path(alg1).resolve())
    mod2 = load_module(Path(alg2).resolve())

    print("[pair] loading structure constants", flush=True)
    t0_load = time.perf_counter()
    c1 = mod1.build_structure_constants(n)
    c2 = mod2.build_structure_constants(n)
    t_load = time.perf_counter() - t0_load

    artifact_cache_mode = str(args_dict.get("artifact_cache_mode", "off"))
    if artifact_cache_mode not in {"on", "off"}:
        artifact_cache_mode = "off"
    artifact_cache_enabled = artifact_cache_mode == "on"
    artifact_cache_path = Path(str(args_dict.get("artifact_cache_path", "isomorphism_artifact_cache.json")))
    artifact_cache = load_artifact_cache(artifact_cache_path) if artifact_cache_enabled else {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    artifact_cache_entries = artifact_cache.get("entries", {}) if isinstance(artifact_cache.get("entries", {}), dict) else {}
    l1_cache: Dict[str, Dict[str, object]] = {}
    artifact_cache_dirty = False
    artifact_cache_stats = {
        "enabled": artifact_cache_enabled,
        "mode": artifact_cache_mode,
        "path": str(artifact_cache_path),
        "rep_hits": 0,
        "rep_misses": 0,
        "profile_hits": 0,
        "profile_misses": 0,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_algo_tag": CACHE_ALGO_TAG,
    }

    print("[pair] analyzing algebra invariants", flush=True)
    t0_analyze = time.perf_counter()
    derivation_mode = str(args_dict.get("derivation_invariant", "auto"))
    if derivation_mode not in {"auto", "on", "off"}:
        derivation_mode = "auto"
    if derivation_mode == "on":
        use_derivation_invariant = True
    elif derivation_mode == "off":
        use_derivation_invariant = False
    else:
        # In same-kernel 2D-extension workflow this invariant is often weak but expensive.
        use_derivation_invariant = not bool(args_dict.get("same_kernel_mode", False))

    pair_parallel_workers = int(args_dict.get("pair_parallel_workers", 1))
    pair_parallel_workers = max(1, min(pair_parallel_workers, 2))
    if artifact_cache_enabled:
        # Keep cache read/write deterministic.
        pair_parallel_workers = 1

    invariant_level = str(args_dict.get("invariant_level", "quick"))
    extension_profile_level = str(args_dict.get("extension_profile_level", "quick"))

    key1 = algebra_cache_key(c1, n, invariant_level, extension_profile_level, use_derivation_invariant)
    key2 = algebra_cache_key(c2, n, invariant_level, extension_profile_level, use_derivation_invariant)

    entry1 = l1_cache.get(key1, artifact_cache_entries.get(key1, {})) if artifact_cache_enabled else {}
    entry2 = l1_cache.get(key2, artifact_cache_entries.get(key2, {})) if artifact_cache_enabled else {}

    rep1 = None
    rep2 = None
    if artifact_cache_enabled and isinstance(entry1, dict) and isinstance(entry1.get("rep"), dict):
        rep1 = deserialize_algebra_report(entry1["rep"])
        if rep1 is not None:
            artifact_cache_stats["rep_hits"] += 1
    if artifact_cache_enabled and isinstance(entry2, dict) and isinstance(entry2.get("rep"), dict):
        rep2 = deserialize_algebra_report(entry2["rep"])
        if rep2 is not None:
            artifact_cache_stats["rep_hits"] += 1

    def _analyze_one(c):
        return analyze_algebra(
            c,
            invariant_level=invariant_level,
            compute_derivation_invariant=use_derivation_invariant,
        )

    if rep1 is None and rep2 is None and pair_parallel_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=pair_parallel_workers) as executor:
            f1 = executor.submit(_analyze_one, c1)
            f2 = executor.submit(_analyze_one, c2)
            rep1 = f1.result()
            rep2 = f2.result()
    else:
        if rep1 is None:
            rep1 = _analyze_one(c1)
        if rep2 is None:
            rep2 = _analyze_one(c2)

    if artifact_cache_enabled:
        if not (isinstance(entry1, dict) and isinstance(entry1.get("rep"), dict)):
            artifact_cache_stats["rep_misses"] += 1
            entry1 = entry1 if isinstance(entry1, dict) else {}
            entry1["rep"] = serialize_algebra_report(rep1)
            artifact_cache_entries[key1] = entry1
            l1_cache[key1] = entry1
            artifact_cache_dirty = True
        if not (isinstance(entry2, dict) and isinstance(entry2.get("rep"), dict)):
            artifact_cache_stats["rep_misses"] += 1
            entry2 = entry2 if isinstance(entry2, dict) else {}
            entry2["rep"] = serialize_algebra_report(rep2)
            artifact_cache_entries[key2] = entry2
            l1_cache[key2] = entry2
            artifact_cache_dirty = True
    t_analyze = time.perf_counter() - t0_analyze

    print("[pair] computing bracket/matrix/action profiles", flush=True)
    t0_profiles = time.perf_counter()
    autn_sig_enabled = not bool(args_dict.get("dev_disable_autn_signature", False))
    profile_modp_mode = str(args_dict.get("profile_modp_mode", "auto"))
    if profile_modp_mode not in {"on", "off", "auto"}:
        profile_modp_mode = "auto"
    try:
        profile_modp_primes = [int(x.strip()) for x in str(args_dict.get("profile_modp_primes", "101,103,107,109,113")).split(",") if x.strip()]
    except Exception:
        profile_modp_primes = [101, 103, 107, 109, 113]

    base_checks = base_invariant_checks(rep1, rep2)
    if profile_modp_mode == "auto":
        strong_align = all(
            bool(base_checks.get(k, False))
            for k in ["dim", "center_dim", "derived_dim", "nilradical_candidate_dim", "extension_dim", "weight_blocks"]
        )
        # Adaptive policy: when strong invariants already align in same-kernel mode,
        # mod-p tends to collide and add overhead; otherwise keep mod-p on for rejection power.
        profile_modp_effective = "off" if (strong_align and bool(args_dict.get("same_kernel_mode", False)) and rep1.extension_dim <= 2) else "on"
    else:
        profile_modp_effective = profile_modp_mode

    def _profiles_one(rep: AlgebraReport):
        return get_profile_context(rep)

    if pair_parallel_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=pair_parallel_workers) as executor:
            f1 = executor.submit(_profiles_one, rep1)
            f2 = executor.submit(_profiles_one, rep2)
            ctx1 = f1.result()
            ctx2 = f2.result()
    else:
        ctx1 = _profiles_one(rep1)
        ctx2 = _profiles_one(rep2)

    if profile_modp_effective == "on":
        bt1_by_prime = []
        bt2_by_prime = []
        ext1_by_prime = []
        ext2_by_prime = []
        bt_mismatch = False
        ext_mismatch = False
        checked_primes = []
        for p in profile_modp_primes:
            b1 = bracket_tensor_profile_modp(rep1.adapted_structure_constants, rep1.n_basis_dim, [p], ctx=ctx1)
            b2 = bracket_tensor_profile_modp(rep2.adapted_structure_constants, rep2.n_basis_dim, [p], ctx=ctx2)
            e1 = extension_action_profile_modp(rep1.adapted_structure_constants, rep1.n_basis_dim, rep1.n, [p], ctx=ctx1)
            e2 = extension_action_profile_modp(rep2.adapted_structure_constants, rep2.n_basis_dim, rep2.n, [p], ctx=ctx2)
            if b1.get("by_prime"):
                bt1_by_prime.extend(b1.get("by_prime", []))
            if b2.get("by_prime"):
                bt2_by_prime.extend(b2.get("by_prime", []))
            if e1.get("by_prime"):
                ext1_by_prime.extend(e1.get("by_prime", []))
            if e2.get("by_prime"):
                ext2_by_prime.extend(e2.get("by_prime", []))
            checked_primes.append(int(p))

            bt_usable = bool(b1.get("available")) and bool(b2.get("available"))
            ext_usable = bool(e1.get("available")) and bool(e2.get("available"))
            if bt_usable and b1.get("signature") != b2.get("signature"):
                bt_mismatch = True
            if ext_usable and e1.get("signature") != e2.get("signature"):
                ext_mismatch = True
            if bt_mismatch or ext_mismatch:
                break

        bt1_modp = {
            "available": bool(bt1_by_prime),
            "reason": None if bt1_by_prime else "no_usable_prime",
            "primes": checked_primes,
            "signature": [(b["p"], b["map_rank"], tuple(b["component_ranks"])) for b in bt1_by_prime],
            "by_prime": bt1_by_prime,
            "stopped_early": bool(bt_mismatch or ext_mismatch),
        }
        bt2_modp = {
            "available": bool(bt2_by_prime),
            "reason": None if bt2_by_prime else "no_usable_prime",
            "primes": checked_primes,
            "signature": [(b["p"], b["map_rank"], tuple(b["component_ranks"])) for b in bt2_by_prime],
            "by_prime": bt2_by_prime,
            "stopped_early": bool(bt_mismatch or ext_mismatch),
        }
        ext1_modp = {
            "available": bool(ext1_by_prime),
            "reason": None if ext1_by_prime else "no_usable_prime",
            "primes": checked_primes,
            "signature": [(b["p"], tuple(b["actions_signature"]), b.get("joint_plane")) for b in ext1_by_prime],
            "by_prime": ext1_by_prime,
            "stopped_early": bool(bt_mismatch or ext_mismatch),
        }
        ext2_modp = {
            "available": bool(ext2_by_prime),
            "reason": None if ext2_by_prime else "no_usable_prime",
            "primes": checked_primes,
            "signature": [(b["p"], tuple(b["actions_signature"]), b.get("joint_plane")) for b in ext2_by_prime],
            "by_prime": ext2_by_prime,
            "stopped_early": bool(bt_mismatch or ext_mismatch),
        }

        bt_modp_usable = bool(bt1_modp.get("available")) and bool(bt2_modp.get("available"))
        ext_modp_usable = bool(ext1_modp.get("available")) and bool(ext2_modp.get("available"))
        modp_profile_checks = {
            "bracket_tensor.modp_signature": (bt1_modp.get("signature") == bt2_modp.get("signature")) if bt_modp_usable else True,
            "extension_action.modp_signature": (ext1_modp.get("signature") == ext2_modp.get("signature")) if ext_modp_usable else True,
        }
    else:
        checked_primes = []
        bt1_modp = {
            "available": False,
            "reason": "disabled",
            "primes": profile_modp_primes,
            "signature": None,
            "by_prime": [],
            "stopped_early": False,
        }
        bt2_modp = dict(bt1_modp)
        ext1_modp = dict(bt1_modp)
        ext2_modp = dict(bt1_modp)
        modp_profile_checks = {
            "bracket_tensor.modp_signature": True,
            "extension_action.modp_signature": True,
        }
    modp_all_pass = all(modp_profile_checks.values()) if profile_modp_effective == "on" else True
    aut1 = None
    aut2 = None
    pencil1 = None
    pencil2 = None

    if profile_modp_effective == "on" and (not modp_all_pass):
        bt1 = {
            "note": "exact profile skipped due to mod-p mismatch",
            "modp_prefilter": bt1_modp,
        }
        bt2 = {
            "note": "exact profile skipped due to mod-p mismatch",
            "modp_prefilter": bt2_modp,
        }
        ext1 = {
            "note": "exact profile skipped due to mod-p mismatch",
            "modp_prefilter": ext1_modp,
        }
        ext2 = {
            "note": "exact profile skipped due to mod-p mismatch",
            "modp_prefilter": ext2_modp,
        }
        mg1 = {"note": "matrix group signature skipped due to mod-p mismatch"}
        mg2 = {"note": "matrix group signature skipped due to mod-p mismatch"}

        checks = base_invariant_checks(rep1, rep2)
        checks.update(modp_profile_checks)
        checks["autn_oriented_signature"] = True if autn_sig_enabled else True
        checks["matrix_group_signature"] = False
        cmp_result = {
            "checks": checks,
            "linear_stage_pass": False,
            "modp_prefilter": {
                "mode": profile_modp_mode,
                "mode_effective": profile_modp_effective,
                "primes": profile_modp_primes,
                "checked_primes": checked_primes,
                "accepted": False,
            },
            "autn_signature": {
                "enabled": autn_sig_enabled,
                "note": "skipped due to mod-p mismatch",
            },
        }
    else:
        can_use_cached_profiles = (
            artifact_cache_enabled
            and isinstance(entry1, dict)
            and isinstance(entry2, dict)
            and isinstance(entry1.get("bt"), dict)
            and isinstance(entry1.get("mg"), dict)
            and isinstance(entry1.get("ext"), dict)
            and isinstance(entry1.get("pencil"), dict)
            and (not autn_sig_enabled or (isinstance(entry1.get("autn"), dict) and isinstance(entry2.get("autn"), dict)))
            and isinstance(entry2.get("bt"), dict)
            and isinstance(entry2.get("mg"), dict)
            and isinstance(entry2.get("ext"), dict)
            and isinstance(entry2.get("pencil"), dict)
        )

        if can_use_cached_profiles:
            bt1 = entry1["bt"]
            mg1 = entry1["mg"]
            ext1 = entry1["ext"]
            pencil1 = entry1["pencil"]
            aut1 = entry1.get("autn") if autn_sig_enabled else None
            bt2 = entry2["bt"]
            mg2 = entry2["mg"]
            ext2 = entry2["ext"]
            pencil2 = entry2["pencil"]
            aut2 = entry2.get("autn") if autn_sig_enabled else None
            artifact_cache_stats["profile_hits"] += 2
        else:
            bt1 = bracket_tensor_profile(rep1.adapted_structure_constants, rep1.n_basis_dim, ctx=ctx1)
            bt2 = bracket_tensor_profile(rep2.adapted_structure_constants, rep2.n_basis_dim, ctx=ctx2)
            mg1 = matrix_group_signature(rep1.adapted_structure_constants, rep1.n_basis_dim, ctx=ctx1)
            mg2 = matrix_group_signature(rep2.adapted_structure_constants, rep2.n_basis_dim, ctx=ctx2)
            ext1 = extension_action_profile(
                rep1.adapted_structure_constants,
                rep1.n_basis_dim,
                rep1.n,
                level=extension_profile_level,
                ctx=ctx1,
            )
            ext2 = extension_action_profile(
                rep2.adapted_structure_constants,
                rep2.n_basis_dim,
                rep2.n,
                level=extension_profile_level,
                ctx=ctx2,
            )
            pencil1 = matrix_pencil_signature(
                rep1.adapted_structure_constants,
                rep1.n_basis_dim,
                ctx=ctx1,
            )
            pencil2 = matrix_pencil_signature(
                rep2.adapted_structure_constants,
                rep2.n_basis_dim,
                ctx=ctx2,
            )
            aut1 = autn_oriented_signature(
                rep1.adapted_structure_constants,
                rep1.n_basis_dim,
                rep1.n,
                rep1.weight_blocks,
                ctx=ctx1,
            ) if autn_sig_enabled else None
            aut2 = autn_oriented_signature(
                rep2.adapted_structure_constants,
                rep2.n_basis_dim,
                rep2.n,
                rep2.weight_blocks,
                ctx=ctx2,
            ) if autn_sig_enabled else None
            if artifact_cache_enabled:
                artifact_cache_stats["profile_misses"] += 2
                entry1 = entry1 if isinstance(entry1, dict) else {}
                entry2 = entry2 if isinstance(entry2, dict) else {}
                entry1["bt"] = to_jsonable(bt1)
                entry1["mg"] = to_jsonable(mg1)
                entry1["ext"] = to_jsonable(ext1)
                entry1["pencil"] = to_jsonable(pencil1)
                if autn_sig_enabled:
                    entry1["autn"] = to_jsonable(aut1)
                entry2["bt"] = to_jsonable(bt2)
                entry2["mg"] = to_jsonable(mg2)
                entry2["ext"] = to_jsonable(ext2)
                entry2["pencil"] = to_jsonable(pencil2)
                if autn_sig_enabled:
                    entry2["autn"] = to_jsonable(aut2)
                artifact_cache_entries[key1] = entry1
                artifact_cache_entries[key2] = entry2
                l1_cache[key1] = entry1
                l1_cache[key2] = entry2
                artifact_cache_dirty = True

        bt1["modp_prefilter"] = bt1_modp
        bt2["modp_prefilter"] = bt2_modp
        ext1["modp_prefilter"] = ext1_modp
        ext2["modp_prefilter"] = ext2_modp

        cmp_result = compare_reports(rep1, rep2, bt1, bt2, ext1, ext2)
        cmp_result["checks"]["autn_oriented_signature"] = (aut1 == aut2) if autn_sig_enabled else True
        cmp_result["checks"]["matrix_pencil_signature"] = (pencil1 == pencil2)
        cmp_result["checks"].update(modp_profile_checks)
        cmp_result["modp_prefilter"] = {
            "mode": profile_modp_mode,
            "mode_effective": profile_modp_effective,
            "primes": profile_modp_primes,
            "checked_primes": checked_primes,
            "accepted": bool(modp_all_pass),
        }
        cmp_result["autn_signature"] = {
            "enabled": autn_sig_enabled,
        }
    t_profiles = time.perf_counter() - t0_profiles
    cmp_result["checks"]["matrix_group_signature"] = mg1.get("weak_signature") == mg2.get("weak_signature")
    cmp_result["linear_stage_pass"] = all(cmp_result["checks"].values())
    print(f"[pair] linear_stage_pass={cmp_result['linear_stage_pass']}", flush=True)

    min_m = min(rep1.n_basis_dim, rep2.n_basis_dim)
    exact_sparse = {"applicable": False, "success": False, "note": "same-kernel mode not enabled"}
    t_exact = 0.0
    if cmp_result["linear_stage_pass"] and bool(args_dict.get("same_kernel_mode", False)):
        print("[pair] trying exact sparse monomial search", flush=True)
        t0_exact = time.perf_counter()
        exact_sparse = exact_sparse_monomial_search(c1, c2, min_m)
        exact_sparse_runtime = time.perf_counter() - t0_exact
        t_exact = float(exact_sparse_runtime)
        exact_sparse["runtime_seconds"] = float(exact_sparse_runtime)
        if exact_sparse.get("success"):
            print("[pair] exact isomorphism found and verified", flush=True)

    if exact_sparse.get("success"):
        system_payload = {"note": "candidate nonlinear system skipped: exact isomorphism already found"}
    elif bool(args_dict.get("skip_system", False)):
        system_payload = {"note": "candidate nonlinear system skipped by --skip-system"}
    elif rep1.n != rep2.n:
        system_payload = {"note": "dimension mismatch; no candidate system generated"}
    elif (not cmp_result["linear_stage_pass"]) and (not bool(args_dict.get("generate_system_on_fail", False))):
        system_payload = {
            "note": "linear stage rejected pair; candidate nonlinear system skipped",
            "hint": "use --generate-system-on-fail to force generation",
        }
    else:
        print("[pair] generating candidate nonlinear system", flush=True)
        system_payload = generate_block_symbolic_system(
            rep1.adapted_structure_constants,
            rep2.adapted_structure_constants,
            min_m,
            rep1.n,
            max_equations=int(args_dict.get("eq_preview", 200)),
            max_pairs=int(args_dict.get("system_max_pairs", 120)),
            nil_weight_labels1=rep1.nil_weight_labels,
            nil_weight_labels2=rep2.nil_weight_labels,
            extension_labels1=rep1.extension_labels,
            extension_labels2=rep2.extension_labels,
            nil_kernel_labels1=rep1.nil_kernel_labels,
            nil_kernel_labels2=rep2.nil_kernel_labels,
            same_kernel_mode=bool(args_dict.get("same_kernel_mode", False)),
        )

    hybrid_payload = {"note": "hybrid solve not requested"}
    if exact_sparse.get("success"):
        hybrid_payload = {"note": "numeric solve skipped: exact sparse search proved isomorphism"}
    elif bool(args_dict.get("hybrid_solve", False)) and isinstance(system_payload, dict) and "_raw" in system_payload:
        print("[pair] running hybrid chain (mod-p + numeric + exact verify)", flush=True)
        raw = system_payload.get("_raw", {})
        eqs = raw.get("equations", [])
        vars_active = raw.get("variables", [])
        try:
            primes = [int(x.strip()) for x in str(args_dict.get("modp_primes", "101,103,107")).split(",") if x.strip()]
        except Exception:
            primes = [101, 103, 107]

        fp = modp_fingerprint(eqs, vars_active, primes)
        num = adaptive_numeric_candidate_search(
            eqs,
            vars_active,
            base_max_vars=int(args_dict.get("numeric_max_vars", 18)),
            base_max_eqs=int(args_dict.get("numeric_max_eqs", 18)),
            base_restarts=int(args_dict.get("numeric_restarts", 8)),
        )
        if num.get("success") and "_raw" in num:
            print("[pair] running exact back-substitution verification", flush=True)
            rat_map = num["_raw"].get("rational_map", {})
            ex = exact_back_substitute_verify(eqs, rat_map, vars_active)
        else:
            print("[pair] numeric stage did not produce a candidate", flush=True)
            ex = {"note": "exact verification skipped (no numeric candidate)"}
        hybrid_payload = {
            "modp_fingerprint": fp,
            "numeric_candidate": {k: v for k, v in num.items() if k != "_raw"},
            "exact_back_substitution": ex,
        }

    if isinstance(system_payload, dict) and "_raw" in system_payload:
        system_payload = dict(system_payload)
        system_payload.pop("_raw", None)

    output = {
        "inputs": {
            "alg1": str(Path(alg1).resolve()),
            "alg2": str(Path(alg2).resolve()),
            "n": n,
        },
        "step1_quick_invariants": {
            "alg1": report_to_dict(rep1),
            "alg2": report_to_dict(rep2),
        },
        "step2_bracket_tensor_on_nilradical_candidate": {
            "alg1": bt1,
            "alg2": bt2,
        },
        "step2b_matrix_group_signature": {
            "alg1": mg1,
            "alg2": mg2,
        },
        "step3_extension_action_profile": {
            "alg1": ext1,
            "alg2": ext2,
        },
        "step3b_autn_oriented_signature": {
            "alg1": aut1 if autn_sig_enabled else {"enabled": False},
            "alg2": aut2 if autn_sig_enabled else {"enabled": False},
            "enabled": autn_sig_enabled,
        },
        "step3c_matrix_pencil_signature": {
            "alg1": pencil1,
            "alg2": pencil2,
            "enabled": True,
        },
        "linear_stage_decision": cmp_result,
        "step4_exact_sparse_monomial_search": exact_sparse,
        "step4b_search_tree_efficiency": {
            "permutations_tested": int(exact_sparse.get("permutations_tested", 0)),
            "full_permutation_count": int(exact_sparse.get("full_permutation_count", 0)) if exact_sparse.get("full_permutation_count") is not None else 0,
            "color_class_permutation_upper_bound": int(exact_sparse.get("color_class_permutation_upper_bound", 0)) if exact_sparse.get("color_class_permutation_upper_bound") is not None else 0,
            "reduction_vs_full_permutations": (
                float(1 - exact_sparse.get("permutations_tested", 0) / exact_sparse.get("full_permutation_count", 1))
                if int(exact_sparse.get("full_permutation_count", 0)) > 0
                else None
            ),
            "reduction_vs_color_class_upper_bound": (
                float(1 - exact_sparse.get("permutations_tested", 0) / exact_sparse.get("color_class_permutation_upper_bound", 1))
                if int(exact_sparse.get("color_class_permutation_upper_bound", 0)) > 0
                else None
            ),
            "baseline_permutations": int(args_dict.get("baseline_permutations", 0)) if args_dict.get("baseline_permutations") else None,
            "reduction_vs_baseline": (
                float(1 - exact_sparse.get("permutations_tested", 0) / int(args_dict.get("baseline_permutations", 1)))
                if args_dict.get("baseline_permutations") and int(args_dict.get("baseline_permutations", 0)) > 0
                else None
            ),
            "exact_search_runtime_seconds": float(exact_sparse.get("runtime_seconds", 0.0)),
        },
        "roi": build_roi_and_suggestions(
            prefilter_reject_rate=(0.0 if cmp_result.get("linear_stage_pass") else 1.0),
            profiles_per_pair_ms=(1000.0 * float(t_profiles)),
            search_tree_reduction=(
                float(1 - exact_sparse.get("permutations_tested", 0) / exact_sparse.get("full_permutation_count", 1))
                if int(exact_sparse.get("full_permutation_count", 0)) > 0
                else None
            ),
        ),
        "step5_candidate_change_of_basis_system": system_payload,
        "step6_hybrid_chain": hybrid_payload,
        "final_decision": {
            "status": "isomorphic" if exact_sparse.get("success") else "undetermined",
            "proved": bool(exact_sparse.get("success")),
            "method": exact_sparse.get("method", "linear filters and hybrid search"),
        },
        "timing": {
            "pipeline_total_seconds": float(time.perf_counter() - t0_pipeline),
            "load_structure_constants_seconds": float(t_load),
            "analyze_invariants_seconds": float(t_analyze),
            "profiles_seconds": float(t_profiles),
            "exact_sparse_search_seconds": float(exact_sparse.get("runtime_seconds", 0.0)),
            "exact_sparse_search_runtime_seconds": float(t_exact),
            "pair_parallel_workers": int(pair_parallel_workers),
        },
        "artifact_cache": artifact_cache_stats,
        "notes": [
            "This script treats [g,g] as a nilradical candidate for the target class (2-step nilpotent kernel + low-dimensional solvable extension).",
            "If linear_stage_pass is false, the pair is rejected as non-isomorphic by linear filters.",
            "If linear_stage_pass is true, use the generated block system for nonlinear solving (Groebner / elimination / numeric-assisted exact checks).",
        ],
    }

    if artifact_cache_enabled:
        artifact_cache["entries"] = artifact_cache_entries
        if artifact_cache_dirty:
            save_artifact_cache(artifact_cache_path, artifact_cache)
    return output


def run_batch_pipeline(args_dict: Dict[str, object]) -> Dict[str, object]:
    t0_batch = time.perf_counter()
    pattern = str(args_dict["batch_glob"])
    files = sorted(Path(p).resolve() for p in glob.glob(pattern) if Path(p).is_file())
    modules = []
    autn_sig_enabled = not bool(args_dict.get("dev_disable_autn_signature", False))

    artifact_cache_mode = str(args_dict.get("artifact_cache_mode", "off"))
    if artifact_cache_mode not in {"on", "off"}:
        artifact_cache_mode = "off"
    artifact_cache_enabled = artifact_cache_mode == "on"
    artifact_cache_path = Path(str(args_dict.get("artifact_cache_path", "isomorphism_artifact_cache.json")))
    artifact_cache = load_artifact_cache(artifact_cache_path) if artifact_cache_enabled else {"version": CACHE_SCHEMA_VERSION, "algo_tag": CACHE_ALGO_TAG, "entries": {}}
    artifact_cache_entries = artifact_cache.get("entries", {}) if isinstance(artifact_cache.get("entries", {}), dict) else {}
    l1_cache: Dict[str, Dict[str, object]] = {}
    artifact_cache_dirty = False
    artifact_cache_stats = {
        "enabled": artifact_cache_enabled,
        "mode": artifact_cache_mode,
        "path": str(artifact_cache_path),
        "rep_hits": 0,
        "rep_misses": 0,
        "profile_hits": 0,
        "profile_misses": 0,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "cache_algo_tag": CACHE_ALGO_TAG,
    }

    invariant_level = str(args_dict.get("invariant_level", "quick"))
    extension_profile_level = str(args_dict.get("extension_profile_level", "quick"))
    pencil_basis_cap_default = int(args_dict.get("pencil_basis_cap", 3))
    pencil_basis_cap_deep = int(args_dict.get("pencil_basis_cap_deep", 6))
    bucket_deepen_min_size = int(args_dict.get("bucket_deepen_min_size", 3))
    bucket_thread_workers = int(args_dict.get("bucket_thread_workers", 0))
    profile_compute_seconds = 0.0
    batch_process_workers = int(args_dict.get("batch_process_workers", 0))
    if batch_process_workers <= 0:
        cpu_count = os.cpu_count() or 2
        # Auto mode: only enable process pool for sufficiently large uncached batches.
        if artifact_cache_enabled or len(files) < 2:
            batch_process_workers_effective = 0
        else:
            batch_process_workers_effective = min(len(files), max(2, min(2, cpu_count // 2)))
    else:
        batch_process_workers_effective = batch_process_workers
    use_process_pool = (batch_process_workers_effective > 1) and (not artifact_cache_enabled)

    if use_process_pool:
        payloads = [
            {
                "path": str(path),
                "n": int(args_dict.get("n", 23)),
                "invariant_level": invariant_level,
                "extension_profile_level": extension_profile_level,
                "same_kernel_mode": bool(args_dict.get("same_kernel_mode", False)),
                "derivation_invariant": str(args_dict.get("derivation_invariant", "auto")),
                "autn_sig_enabled": autn_sig_enabled,
            }
            for path in files
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=batch_process_workers_effective) as executor:
            for item in executor.map(_batch_process_worker, payloads):
                if not item:
                    continue
                rep = deserialize_algebra_report(item["rep"])
                if rep is None:
                    continue
                modules.append(
                    {
                        "path": str(item["path"]),
                        "n": int(item["n"]),
                        "rep": rep,
                        "key": str(item["key"]),
                        "bt": item["bt"],
                        "mg": item["mg"],
                        "ext": item["ext"],
                        "autn": item.get("autn"),
                        "pencil": matrix_pencil_signature(rep.adapted_structure_constants, rep.n_basis_dim, basis_cap=pencil_basis_cap_default),
                    }
                )
    else:
        for path in files:
            try:
                module = load_module(path)
            except Exception:
                continue
            n = int(args_dict.get("n", 23))
            try:
                c = module.build_structure_constants(n)
            except Exception:
                continue
            derivation_mode = str(args_dict.get("derivation_invariant", "auto"))
            if derivation_mode == "on":
                use_derivation_invariant = True
            elif derivation_mode == "off":
                use_derivation_invariant = False
            else:
                use_derivation_invariant = not bool(args_dict.get("same_kernel_mode", False))
            key = algebra_cache_key(c, n, invariant_level, extension_profile_level, use_derivation_invariant)
            entry = l1_cache.get(key, artifact_cache_entries.get(key, {})) if artifact_cache_enabled else {}

            rep = None
            if artifact_cache_enabled and isinstance(entry, dict) and isinstance(entry.get("rep"), dict):
                rep = deserialize_algebra_report(entry["rep"])
                if rep is not None:
                    artifact_cache_stats["rep_hits"] += 1
            if rep is None:
                rep = analyze_algebra(
                    c,
                    invariant_level=invariant_level,
                    compute_derivation_invariant=use_derivation_invariant,
                )
                if artifact_cache_enabled:
                    artifact_cache_stats["rep_misses"] += 1
                    entry = entry if isinstance(entry, dict) else {}
                    entry["rep"] = serialize_algebra_report(rep)
                    artifact_cache_entries[key] = entry
                    l1_cache[key] = entry
                    artifact_cache_dirty = True
            modules.append(
                {
                    "path": str(path),
                    "n": n,
                    "key": key,
                    "rep": rep,
                    "bt": None,
                    "mg": None,
                    "ext": None,
                    "autn": None,
                    "pencil": None,
                    "pencil_deep": None,
                }
            )

    def ensure_profiles(item: Dict[str, object], deep: bool = False) -> None:
        nonlocal profile_compute_seconds, artifact_cache_dirty
        if item.get("bt") is not None and item.get("mg") is not None and item.get("ext") is not None and item.get("pencil") is not None:
            if deep and item.get("pencil_deep") is None:
                rep = item["rep"]
                ctx = get_profile_context(rep)
                t0p = time.perf_counter()
                item["pencil_deep"] = matrix_pencil_signature(
                    rep.adapted_structure_constants,
                    rep.n_basis_dim,
                    ctx=ctx,
                    basis_cap=pencil_basis_cap_deep,
                )
                profile_compute_seconds += time.perf_counter() - t0p
            return

        if artifact_cache_enabled:
            entry = l1_cache.get(item["key"], artifact_cache_entries.get(item["key"], {}))
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("bt"), dict)
                and isinstance(entry.get("mg"), dict)
                and isinstance(entry.get("ext"), dict)
                and isinstance(entry.get("pencil"), dict)
                and (not autn_sig_enabled or isinstance(entry.get("autn"), dict))
            ):
                item["bt"] = entry["bt"]
                item["mg"] = entry["mg"]
                item["ext"] = entry["ext"]
                item["pencil"] = entry["pencil"]
                item["autn"] = entry.get("autn") if autn_sig_enabled else None
                artifact_cache_stats["profile_hits"] += 1
                if deep and isinstance(entry.get("pencil_deep"), dict):
                    item["pencil_deep"] = entry["pencil_deep"]
                elif deep:
                    rep = item["rep"]
                    ctx = get_profile_context(rep)
                    t0p = time.perf_counter()
                    item["pencil_deep"] = matrix_pencil_signature(
                        rep.adapted_structure_constants,
                        rep.n_basis_dim,
                        ctx=ctx,
                        basis_cap=pencil_basis_cap_deep,
                    )
                    profile_compute_seconds += time.perf_counter() - t0p
                    entry["pencil_deep"] = to_jsonable(item["pencil_deep"])
                    artifact_cache_entries[item["key"]] = entry
                    l1_cache[item["key"]] = entry
                    artifact_cache_dirty = True
                return

        rep = item["rep"]
        ctx = get_profile_context(rep)
        t0p = time.perf_counter()
        bt = bracket_tensor_profile(rep.adapted_structure_constants, rep.n_basis_dim, ctx=ctx)
        mg = matrix_group_signature(rep.adapted_structure_constants, rep.n_basis_dim, ctx=ctx)
        ext = extension_action_profile(
            rep.adapted_structure_constants,
            rep.n_basis_dim,
            rep.n,
            level=extension_profile_level,
            ctx=ctx,
        )
        autn = autn_oriented_signature(
            rep.adapted_structure_constants,
            rep.n_basis_dim,
            rep.n,
            rep.weight_blocks,
            ctx=ctx,
        ) if autn_sig_enabled else None
        pencil = matrix_pencil_signature(rep.adapted_structure_constants, rep.n_basis_dim, ctx=ctx, basis_cap=pencil_basis_cap_default)
        profile_compute_seconds += time.perf_counter() - t0p
        item["bt"] = bt
        item["mg"] = mg
        item["ext"] = ext
        item["autn"] = autn
        item["pencil"] = pencil
        if deep:
            t1p = time.perf_counter()
            item["pencil_deep"] = matrix_pencil_signature(
                rep.adapted_structure_constants,
                rep.n_basis_dim,
                ctx=ctx,
                basis_cap=pencil_basis_cap_deep,
            )
            profile_compute_seconds += time.perf_counter() - t1p

        if artifact_cache_enabled:
            artifact_cache_stats["profile_misses"] += 1
            entry = l1_cache.get(item["key"], artifact_cache_entries.get(item["key"], {}))
            entry = entry if isinstance(entry, dict) else {}
            entry["bt"] = to_jsonable(bt)
            entry["mg"] = to_jsonable(mg)
            entry["ext"] = to_jsonable(ext)
            entry["pencil"] = to_jsonable(pencil)
            if item.get("pencil_deep") is not None:
                entry["pencil_deep"] = to_jsonable(item["pencil_deep"])
            if autn_sig_enabled:
                entry["autn"] = to_jsonable(autn)
            artifact_cache_entries[item["key"]] = entry
            l1_cache[item["key"]] = entry
            artifact_cache_dirty = True
    # Step 1: light-signature bucketing.
    light_buckets: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for item in modules:
        rep = item["rep"]
        h_light = light_invariant_hash(rep)
        light_buckets.setdefault((int(item["n"]), h_light), []).append(item)

    # Step 2: collision-rate assessment.
    light_bucket_sizes = [len(v) for v in light_buckets.values()]
    collision_bucket_count = sum(1 for s in light_bucket_sizes if s > 1)
    collision_pair_count = sum(s * (s - 1) // 2 for s in light_bucket_sizes if s > 1)

    # Step 3: secondary deepening on high-collision buckets.
    final_groups: List[Tuple[int, str, List[Dict[str, object]], bool]] = []
    deepened_bucket_count = 0
    for (dim, hlight), items in sorted(light_buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(items) <= 1:
            final_groups.append((dim, hlight, items, False))
            continue
        need_deepen = len(items) >= bucket_deepen_min_size
        if need_deepen:
            deepened_bucket_count += 1
            split: Dict[str, List[Dict[str, object]]] = {}
            for it in items:
                ensure_profiles(it, deep=True)
                p = it.get("pencil_deep") if isinstance(it.get("pencil_deep"), dict) else it.get("pencil")
                pkey = json.dumps(to_jsonable(p), sort_keys=True, ensure_ascii=False)
                split.setdefault(pkey, []).append(it)
            for idx, sub in enumerate(split.values(), start=1):
                final_groups.append((dim, f"{hlight}:deep:{idx}", sub, True))
        else:
            for it in items:
                ensure_profiles(it, deep=False)
            final_groups.append((dim, hlight, items, False))

    pair_results = []
    def compare_pair(payload: Tuple[int, str, Dict[str, object], Dict[str, object], bool]) -> Dict[str, object]:
        dim, hsig, a, b, used_deepen = payload
        cmp_result = compare_reports(a["rep"], b["rep"], a["bt"], b["bt"], a["ext"], b["ext"])
        cmp_result["checks"]["autn_oriented_signature"] = (a.get("autn") == b.get("autn")) if autn_sig_enabled else True
        cmp_result["checks"]["matrix_pencil_signature"] = (a.get("pencil") == b.get("pencil"))
        cmp_result["checks"]["matrix_group_signature"] = a["mg"].get("weak_signature") == b["mg"].get("weak_signature")
        cmp_result["linear_stage_pass"] = all(cmp_result["checks"].values())
        return {
            "dim": dim,
            "bucket_hash": hsig,
            "used_secondary_deepening": used_deepen,
            "alg1": a["path"],
            "alg2": b["path"],
            "linear_stage_pass": cmp_result["linear_stage_pass"],
            "checks": cmp_result["checks"],
            "weight_block_signature_1": weight_block_signature(a["rep"]),
            "weight_block_signature_2": weight_block_signature(b["rep"]),
            "matrix_group_weak_signature_1": a["mg"].get("weak_signature"),
            "matrix_group_weak_signature_2": b["mg"].get("weak_signature"),
            "autn_signature_enabled": autn_sig_enabled,
            "batch_process_workers": int(batch_process_workers_effective),
        }

    for dim, hsig, items, used_deepen in final_groups:
        if len(items) < 2:
            continue
        tasks = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                tasks.append((dim, hsig, items[i], items[j], used_deepen))

        if bucket_thread_workers <= 0:
            cpu_count = os.cpu_count() or 2
            bucket_workers_eff = max(1, min(4, cpu_count // 2))
        else:
            bucket_workers_eff = bucket_thread_workers

        # Step 4: bucket-local parallel compare for larger buckets, sequential for tiny buckets.
        if len(tasks) >= 6 and batch_process_workers_effective > 1 and bucket_workers_eff > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=bucket_workers_eff) as executor:
                pair_results.extend(list(executor.map(compare_pair, tasks)))
        else:
            pair_results.extend(compare_pair(t) for t in tasks)

    pair_count = len(pair_results)
    reject_count = sum(1 for r in pair_results if not bool(r.get("linear_stage_pass", False)))
    prefilter_reject_rate = (reject_count / pair_count) if pair_count > 0 else None
    profiles_per_pair_ms = (1000.0 * profile_compute_seconds / pair_count) if (pair_count > 0 and profile_compute_seconds > 0) else None
    roi = build_roi_and_suggestions(
        prefilter_reject_rate=prefilter_reject_rate,
        profiles_per_pair_ms=profiles_per_pair_ms,
        search_tree_reduction=None,
    )

    out = {
        "batch_glob": pattern,
        "file_count": len(files),
        "loaded_count": len(modules),
        "dimensions": sorted({k[0] for k in light_buckets.keys()}),
        "bucket_count": len(final_groups),
        "light_bucket_count": len(light_buckets),
        "collision_bucket_count": collision_bucket_count,
        "collision_pair_count": int(collision_pair_count),
        "deepened_bucket_count": int(deepened_bucket_count),
        "pair_count": len(pair_results),
        "pairs": pair_results,
        "roi": roi,
        "artifact_cache": artifact_cache_stats,
        "timing": {
            "pipeline_total_seconds": float(time.perf_counter() - t0_batch),
            "profiles_total_seconds": (float(profile_compute_seconds) if profile_compute_seconds > 0 else None),
            "batch_process_workers": int(batch_process_workers_effective),
            "batch_process_workers_requested": int(batch_process_workers),
            "used_process_pool": bool(use_process_pool),
        },
    }

    if artifact_cache_enabled:
        artifact_cache["entries"] = artifact_cache_entries
        if artifact_cache_dirty:
            save_artifact_cache(artifact_cache_path, artifact_cache)
    return out


def main():
    parser = argparse.ArgumentParser(description="Pipeline for solvable Lie algebra isomorphism filtering.")
    parser.add_argument("--alg1", help="Path to first Python file defining build_structure_constants(n)")
    parser.add_argument("--alg2", help="Path to second Python file defining build_structure_constants(n)")
    parser.add_argument("--batch-glob", help="Glob pattern for batch mode, e.g. '*.py'")
    parser.add_argument("--n", type=int, default=23, help="Lie algebra dimension")
    parser.add_argument("--out", default="isomorphism_report.json", help="Output report JSON path")
    parser.add_argument("--eq-preview", type=int, default=200, help="How many polynomial equations to preview")
    parser.add_argument(
        "--system-max-pairs",
        type=int,
        default=120,
        help="Maximum bracket pairs used when generating candidate nonlinear system",
    )
    parser.add_argument(
        "--extension-profile-level",
        choices=["quick", "full"],
        default="quick",
        help="quick avoids expensive Jordan/charpoly computations; full computes stronger but slower invariants",
    )
    parser.add_argument(
        "--invariant-level",
        choices=["quick", "full"],
        default="quick",
        help="quick computes only low-cost invariants; full includes deeper derived/lower-central series",
    )
    parser.add_argument(
        "--skip-system",
        action="store_true",
        help="Skip candidate nonlinear system generation and output linear-stage report only",
    )
    parser.add_argument(
        "--generate-system-on-fail",
        action="store_true",
        help="Generate candidate nonlinear system even when linear stage rejects the pair",
    )
    parser.add_argument(
        "--same-kernel-mode",
        action="store_true",
        help="Assume both solvable algebras are extensions of the same 2-step nilpotent kernel; add stricter Xn block constraints",
    )
    parser.add_argument(
        "--hybrid-solve",
        action="store_true",
        help="Run hybrid chain: mod-p fingerprint + numeric candidate + exact back-substitution",
    )
    parser.add_argument(
        "--modp-primes",
        default="101,103,107",
        help="Comma-separated small primes for mod-p fingerprint",
    )
    parser.add_argument(
        "--numeric-max-vars",
        type=int,
        default=18,
        help="Max variables in reduced subsystem for numeric candidate search",
    )
    parser.add_argument(
        "--numeric-max-eqs",
        type=int,
        default=18,
        help="Max equations in reduced subsystem for numeric candidate search",
    )
    parser.add_argument(
        "--numeric-restarts",
        type=int,
        default=8,
        help="Random restarts for numeric candidate search",
    )
    parser.add_argument(
        "--baseline-permutations",
        type=int,
        default=0,
        help="Optional baseline permutations_tested value for reporting reduction ratio",
    )
    parser.add_argument(
        "--derivation-invariant",
        choices=["auto", "on", "off"],
        default="auto",
        help="Control whether derivation algebra dimension is computed as an extra filter (auto defaults to off in same-kernel mode).",
    )
    parser.add_argument(
        "--pair-parallel-workers",
        type=int,
        default=1,
        help="Parallel workers for pair-mode invariant/profile stages (1 disables parallelism).",
    )
    parser.add_argument(
        "--profile-modp-mode",
        choices=["on", "off", "auto"],
        default="auto",
        help="Use mod-p profile signatures as prefilter: on/off/auto (adaptive).",
    )
    parser.add_argument(
        "--profile-modp-primes",
        default="101,103,107,109,113",
        help="Comma-separated primes used by mod-p profile prefilter.",
    )
    parser.add_argument(
        "--artifact-cache-mode",
        choices=["on", "off"],
        default="off",
        help="Step-4 cache switch: cache per-algebra analyze/profile artifacts across runs.",
    )
    parser.add_argument(
        "--artifact-cache-path",
        default="isomorphism_artifact_cache.json",
        help="Path to the step-4 artifact cache file.",
    )
    parser.add_argument(
        "--batch-process-workers",
        type=int,
        default=0,
        help="Batch process workers (>1 forces process pool, 0 uses auto recommendation, 1 disables process pool).",
    )
    parser.add_argument(
        "--bucket-deepen-min-size",
        type=int,
        default=3,
        help="Enable secondary deep signature inside a light bucket when bucket size >= this threshold.",
    )
    parser.add_argument(
        "--bucket-thread-workers",
        type=int,
        default=0,
        help="Per-bucket compare workers in threaded mode (0 for auto).",
    )
    parser.add_argument(
        "--pencil-basis-cap",
        type=int,
        default=3,
        help="Default matrix-pencil basis sampling cap.",
    )
    parser.add_argument(
        "--pencil-basis-cap-deep",
        type=int,
        default=6,
        help="Matrix-pencil basis cap used for secondary deepening in high-collision buckets.",
    )
    parser.add_argument(
        "--dev-disable-autn-signature",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    args_dict = vars(args)
    if args.batch_glob:
        output = run_batch_pipeline(args_dict)
    else:
        if not args.alg1 or not args.alg2:
            raise SystemExit("single-pair mode requires --alg1 and --alg2")
        output = run_pair_pipeline(args_dict)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(to_jsonable(output), indent=2, ensure_ascii=False), encoding="utf-8")

    next_run_artifacts = write_next_run_templates(args_dict, output, out_path)

    print(f"Wrote report: {out_path}")
    if next_run_artifacts:
        print(f"Wrote next-run template: {next_run_artifacts['template_path']}")
        print(f"Wrote next-run suggestions: {next_run_artifacts['suggestions_path']}")
    if not args.batch_glob:
        cmp_result = output.get("linear_stage_decision", {})
        system_payload = output.get("step5_candidate_change_of_basis_system", {})
        print(f"Linear stage pass: {cmp_result.get('linear_stage_pass')}")
        print(f"Candidate equations (total): {system_payload.get('equation_count_total', 0)}")
        print(f"Final decision: {output.get('final_decision', {}).get('status', 'undetermined')}")


if __name__ == "__main__":
    main()
