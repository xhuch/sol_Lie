import json
import itertools
import sympy as sp
from fractions import Fraction

# Load structure constants builder from local compute_derivations.py
# When running as: python invariant_extractor.py, sys.path[0] is this script's dir.
from compute_derivations import build_structure_constants


def center_from_c(n, c):
    # Solve for z = (z_i) with [z, e_j] = 0 for all j
    z_vars = [sp.Symbol(f'z{i+1}') for i in range(n)]
    eqs = []
    for j in range(n):
        for k in range(n):
            eq = sum(z_vars[i] * c[i][j][k] for i in range(n))
            eqs.append(sp.simplify(eq))
    A, b = sp.linear_eq_to_matrix(eqs, z_vars)
    ns = A.nullspace()
    center_basis = []
    for v in ns:
        vec = [sp.simplify(v[i]) for i in range(n)]
        center_basis.append(vec)
    return center_basis


def span_from_brackets(n, c):
    # collect all bracket vectors [e_i, e_j] as column vectors
    cols = []
    for i in range(n):
        for j in range(i+1, n):
            v = [c[i][j][k] for k in range(n)]
            if any(x != 0 for x in v):
                cols.append([sp.simplify(x) for x in v])
    if not cols:
        return []
    M = sp.Matrix(cols).T
    # compute column space basis (as vectors of length n)
    cs = M.columnspace()
    basis = [[sp.simplify(x) for x in vec] for vec in cs]
    return basis


def lower_central_series(n, c):
    # Represent subspaces as list of basis vectors (length-n lists)
    # Start with g = span of standard basis (identity)
    # We'll compute derived = span of brackets of g and previous
    # For efficiency use symbolic linear algebra on matrices
    E = [sp.Matrix([[1 if i==j else 0] for i in range(n)]) for j in range(n)]
    # basis vectors as matrices column vectors
    current = [v for v in E]
    series_dims = [n]
    for step in range(1, n+1):
        # compute [g, current] span
        cols = []
        for i in range(n):
            ei = [0]*n; ei[i]=1
            for vmat in current:
                # vmat is n x 1
                v = [sp.simplify(vmat[r,0]) for r in range(n)]
                for j in range(n):
                    coeff = v[j]
                    if coeff == 0:
                        continue
                    # [e_i, sum_j v_j e_j] = sum_j v_j [e_i, e_j]
                    b = [sp.simplify(sum(v[j]*c[i][j][k] for j in range(n))) for k in range(n)]
                    if any(x != 0 for x in b):
                        cols.append(b)
        if not cols:
            series_dims.append(0)
            break
        M = sp.Matrix(cols).T
        cs = M.columnspace()
        dim = len(cs)
        series_dims.append(dim)
        # if stabilized to zero or same dim as previous, stop
        if dim == 0:
            break
        # set current to columnspace basis as n x 1 matrices
        current = [sp.Matrix([[cs[i][r]] for r in range(n)]).col_insert(0, sp.Matrix([[0]])) for i in range(len(cs))]  # dummy reshape
        # The above is clumsy; instead keep current as vectors lists
        current = [sp.Matrix([[cs_i[r]] for r in range(n)]) for cs_i in cs]
        if dim == series_dims[-2]:
            break
    return series_dims


def bracket_map_V_wedge_V_to_Z(n, c, center_basis):
    # center_basis: list of basis vectors (lists length n)
    if not center_basis:
        return {'Z_dim':0, 'map_rank':0, 'Z_basis_indices':[]}
    # Build Z basis matrix (n x zdim)
    Zmat = sp.Matrix([[center_basis[j][i] for j in range(len(center_basis))] for i in range(n)])
    zdim = Zmat.cols
    # choose V indices as complement of span(Z) — compute projection test: a standard basis e_i is in Z span?
    # We'll find which standard basis vectors are in span(Z) by solving Zmat * alpha = e_i
    Zcols = Zmat
    V_idx = []
    Z_idx = []
    for i in range(n):
        e = sp.Matrix([1 if r==i else 0 for r in range(n)])
        try:
            sol = Zcols.gauss_jordan_solve(e)
            # if solvable, e in Z span
            Z_idx.append(i)
        except Exception:
            V_idx.append(i)
    # Now map pairs from V wedge V to coordinates in Z basis
    pairs = [(i,j) for i in V_idx for j in V_idx if i<j]
    if not pairs:
        return {'Z_dim':zdim, 'map_rank':0, 'Z_basis_indices':Z_idx}
    M = sp.zeros(zdim, len(pairs))
    for col, (i,j) in enumerate(pairs):
        bracket = sp.Matrix([c[i][j][k] for k in range(n)])
        # find coordinates in Z basis by solving Zcols * x = bracket (least squares exact)
        # Use linear_eq_to_matrix
        A, b = sp.linear_eq_to_matrix([bracket[r,0] for r in range(n)], [sp.Symbol(f'a{t}') for t in range(zdim)])
        # but easier: compute coords = (Zcols.T*Zcols)^-1 * Zcols.T * bracket if invertible
        if zdim>0:
            try:
                coords = (Zcols.T*Zcols).inv() * Zcols.T * bracket
            except Exception:
                # fallback to nullspace-based solve
                A2, b2 = sp.linear_eq_to_matrix([sp.Sum(sp.Symbol(f'a{t}')*Zcols[r,t], (t,0,zdim-1)) - bracket[r,0] for r in range(n)], [sp.Symbol(f'a{t}') for t in range(zdim)])
                coords = sp.Matrix([0]*zdim)
            for r in range(zdim):
                M[r, col] = sp.simplify(coords[r])
    rank = M.rank()
    return {'Z_dim': zdim, 'map_rank': rank, 'Z_basis_indices': Z_idx, 'pairs_count': len(pairs)}


def compute_invariants(n, c):
    center_basis = center_from_c(n, c)
    center_dim = len(center_basis)
    derived_basis = span_from_brackets(n, c)
    derived_dim = len(derived_basis)
    # lower central series dims (approx)
    # For now compute derived dimension iteratively once
    lcs = [n, derived_dim]
    bracket_map_info = bracket_map_V_wedge_V_to_Z(n, c, center_basis)
    return {
        'n': n,
        'center_dim': center_dim,
        'derived_dim': derived_dim,
        'lower_central_series_dims': lcs,
        'bracket_map': {k: int(v) if isinstance(v, (int, sp.Integer)) else v for k,v in bracket_map_info.items()}
    }


def main():
    n = 23
    c = build_structure_constants(n)
    inv = compute_invariants(n, c)
    with open('invariants_A.json', 'w', encoding='utf-8') as f:
        json.dump(inv, f, indent=2)
    print('Wrote invariants_A.json')

if __name__ == '__main__':
    main()
