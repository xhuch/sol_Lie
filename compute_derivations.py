from fractions import Fraction
import sympy as sp

def R(x):
    if isinstance(x, Fraction):
        return sp.Rational(x.numerator, x.denominator)
    return sp.Rational(x)

def build_structure_constants(n):
    # c[i][j][k] corresponds to [e_{i+1}, e_{j+1}] = sum_k c[i][j][k] e_{k+1}
    c = [[[sp.Rational(0) for _ in range(n)] for _ in range(n)] for _ in range(n)]

    def add(i,j,k,coeff=1):
        # indices provided 1-based, convert to 0-based
        a,b,cidx = i-1, j-1, k-1
        coeff = R(coeff)
        c[a][b][cidx] += coeff
        c[b][a][cidx] -= coeff

    # --- bracket relations provided by user ---
    add(1,6,18)
    add(1,7,19)
    add(2,3,21)
    add(2,4,13)
    add(2,7,12)
    add(3,4,14)
    add(4,5,9)
    add(4,7,15)
    add(5,6,10)
    add(5,7,17)
    add(5,8,16)
    add(6,7,11)
    add(7,8,20)

    # actions by e22 and e23 (diagonal-type commutators)
    for i in range(1,9):
        add(i,22,i, Fraction(1,2))  # [e_i,e_22] = 1/2 e_i
    for i in range(1,9):
        add(i,23,i, 1)  # [e_i,e_23] = e_i

    add(9,23,9,2)
    add(10,23,10,2)
    add(11,22,11,1)
    add(12,23,12,2)
    add(13,22,13,1)
    add(14,22,14,1)
    add(15,22,15,1)
    add(16,23,16,2)
    add(17,22,17,1)
    add(18,23,18,2)
    add(19,22,19,1)
    add(20,22,20,1)
    add(21,23,21,2)

    return c

def compute_derivations(n, c):
    # Unknowns D^k_i (D(e_i) = sum_k D^k_i e_k). We'll flatten unknowns in column-major: (k,i)
    D_vars = [sp.Symbol(f'd{r+1}_{s+1}') for r in range(n) for s in range(n)]

    eqs = []

    # Precompute nonzero (i,j) pairs to reduce work
    pairs = [(i,j) for i in range(n) for j in range(n) if any(c[i][j][k] != 0 for k in range(n))]

    for i,j in pairs:
        for k in range(n):
            # left: sum_m c_{ij}^m * D^k_m
            left = sum(c[i][j][m] * D_vars[k*n + m] for m in range(n))
            # right: sum_p D^p_i * c_{pj}^k + sum_q D^q_j * c_{iq}^k
            right1 = sum(D_vars[p*n + i] * c[p][j][k] for p in range(n))
            right2 = sum(D_vars[q*n + j] * c[i][q][k] for q in range(n))
            eqs.append(sp.simplify(left - right1 - right2))

    # Convert to matrix form and compute nullspace (solution space for D)
    if not eqs:
        # no relations -> all linear maps are derivations
        M = sp.zeros(0, n*n)
        ns = [sp.eye(n*n).col(i) for i in range(n*n)]
    else:
        A, b = sp.linear_eq_to_matrix(eqs, D_vars)
        ns = A.nullspace()

    # Format nullspace vectors as matrices
    derivations = []
    for v in ns:
        Mv = sp.Matrix(n, n, lambda r, s: sp.simplify(v[r*n + s]))
        derivations.append(Mv)

    return derivations


def main():
    n = 23
    c = build_structure_constants(n)
    derivs = compute_derivations(n, c)
    print(f"Dimension of derivation algebra: {len(derivs)}")
    for idx, M in enumerate(derivs, 1):
        print(f"\nDerivation basis #{idx} (matrix {n}x{n}):")
        sp.pprint(M)

if __name__ == '__main__':
    main()
