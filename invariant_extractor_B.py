import json
import sympy as sp
from fractions import Fraction

def R(x):
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    return sp.Rational(x)

def build_structure_constants(n):
    c = [[[sp.Rational(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]
    def add(i,j,k,coeff=1):
        a,b,cidx = i-1, j-1, k-1
        coeff = R(coeff)
        c[a][b][cidx] += coeff
        c[b][a][cidx] -= coeff

    # brackets for Algebra B (user-supplied)
    add(1,3,13)
    add(1,4,21)
    add(1,6,14)
    add(2,5,18)
    add(2,6,19)
    add(3,4,12)
    add(4,5,11)
    add(4,6,20)
    add(5,6,17)
    add(5,8,10)
    add(6,7,15)
    add(6,8,16)
    add(7,8,9)

    for i in range(1,9):
        add(i,22,i, Fraction(1,2))
    for i in range(1,9):
        add(i,23,i, 1)

    add(9,23,9,2)
    add(10,23,10,2)
    add(11,23,11,2)
    add(12,22,12,1)
    add(13,23,13,2)
    add(14,23,14,2)
    add(15,22,15,1)
    add(16,22,16,1)
    add(17,22,17,1)
    add(18,23,18,2)
    add(19,22,19,1)
    add(20,22,20,1)
    add(21,22,21,1)

    return c

# Reuse invariant routines (same approach as extractor A)

def center_from_c(n, c):
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
    cols = []
    for i in range(n):
        for j in range(i+1, n):
            v = [c[i][j][k] for k in range(n)]
            if any(x != 0 for x in v):
                cols.append([sp.simplify(x) for x in v])
    if not cols:
        return []
    M = sp.Matrix(cols).T
    cs = M.columnspace()
    basis = [[sp.simplify(x) for x in vec] for vec in cs]
    return basis


def bracket_map_V_wedge_V_to_Z(n, c, center_basis):
    if not center_basis:
        return {'Z_dim':0, 'map_rank':0, 'Z_basis_indices':[]}
    Zmat = sp.Matrix([[center_basis[j][i] for j in range(len(center_basis))] for i in range(n)])
    zdim = Zmat.cols
    V_idx = []
    Z_idx = []
    for i in range(n):
        e = sp.Matrix([1 if r==i else 0 for r in range(n)])
        try:
            sol = Zmat.gauss_jordan_solve(e)
            Z_idx.append(i)
        except Exception:
            V_idx.append(i)
    pairs = [(i,j) for i in V_idx for j in V_idx if i<j]
    if not pairs:
        return {'Z_dim':zdim, 'map_rank':0, 'Z_basis_indices':Z_idx}
    M = sp.zeros(zdim, len(pairs))
    for col, (i,j) in enumerate(pairs):
        bracket = sp.Matrix([c[i][j][k] for k in range(n)])
        if zdim>0:
            try:
                coords = (Zmat.T*Zmat).inv() * Zmat.T * bracket
            except Exception:
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
    with open('invariants_B.json', 'w', encoding='utf-8') as f:
        json.dump(inv, f, indent=2)
    print('Wrote invariants_B.json')

if __name__ == '__main__':
    main()
