"""Exact spectral identities and high-precision Lyapunov coefficients.

Run: python Codigo/adaptativo_hopf.py
SymPy provides exact symbolic assertions and an exact algebraic l1 sign certificate;
mpmath independently evaluates l1 at 60/100 digits for numerical presentation.
Numerical agreement alone is NOT interval certification.
Convention: q[0]=1, conjugate(p).T*q=1, F=J*u+B(u,u)/2+C(u,u,u)/6+...
"""
import sympy as sp
import mpmath as mp


def exact_checks():
    k, lam = sp.symbols("k lambda")
    J = sp.Matrix([[-1, -1, -10], [4, -1, -1], [k / 10, 4 * k / 5, -k]])
    P = sp.expand(J.charpoly(lam).as_expr())
    expected = lam**3 + (k+2)*lam**2 + (5+sp.Rational(19, 5)*k)*lam + sp.Rational(387, 10)*k
    assert sp.expand(P-expected) == 0
    H = sp.expand((k+2)*(5+sp.Rational(19, 5)*k)-sp.Rational(387, 10)*k)
    roots = [(261-sign*sp.sqrt(52921))/76 for sign in [1, -1]]
    for root in roots:
        assert sp.simplify(H.subs(k, root)) == 0
    # Derive all multilinear forms directly from the rational vector field.
    x, y, z = sp.symbols("x y z")
    F = sp.Matrix([x*(12-x-y-10*z), y*(-2+4*x-y-z), k*(10*(x+8*y)/(81+x+8*y)-z)])
    assert F.subs({x: 1, y: 1, z: 1}) == sp.zeros(3, 1)
    assert F.jacobian([x, y, z]).subs({x: 1, y: 1, z: 1}) == J
    assert sp.diff(F[2], x, 2).subs({x: 1, y: 1, z: 1}) == -k/450
    assert sp.diff(F[2], x, 3).subs({x: 1, y: 1, z: 1}) == k/13500
    print("SymPy", sp.__version__, "mpmath", mp.__version__)
    print("P =", P)
    print("H =", H)
    print("Exact thresholds:", roots)


def evaluate(dps, branch):
    mp.mp.dps = dps
    k = (261 + branch * mp.sqrt(52921)) / 76
    w = mp.sqrt(5+mp.mpf(19)*k/5)
    J = mp.matrix([[-1, -1, -10], [4, -1, -1], [k/10, 4*k/5, -k]])
    qy = (41+mp.j*w)/(9+10*mp.j*w)
    q = mp.matrix([1, qy, 4-(1+mp.j*w)*qy])
    pv = mp.lu_solve(mp.matrix([[-1+mp.j*w, 4], [-1, -1+mp.j*w]]), mp.matrix([-k/10, -4*k/5]))
    p0 = mp.matrix([pv[0], pv[1], 1])

    def inner(u, v):
        return sum(mp.conj(u[i])*v[i] for i in range(3))

    p = p0 / mp.conj(inner(p0, q))

    def B(u, v):
        return mp.matrix([
            -2*u[0]*v[0]-(u[0]*v[1]+u[1]*v[0])-10*(u[0]*v[2]+u[2]*v[0]),
            4*(u[0]*v[1]+u[1]*v[0])-2*u[1]*v[1]-(u[1]*v[2]+u[2]*v[1]),
            -k*(u[0]+8*u[1])*(v[0]+8*v[1])/450,
        ])

    def C(u, v, t):
        return mp.matrix([0, 0, k*(u[0]+8*u[1])*(v[0]+8*v[1])*(t[0]+8*t[1])/13500])

    qb = q.conjugate()
    h11 = mp.lu_solve(J, B(q, qb))
    h20 = mp.lu_solve(2*mp.j*w*mp.eye(3)-J, B(q, q))
    G21 = inner(p, C(q, q, qb)-2*B(q, h11)+B(qb, h20))
    l1 = mp.re(G21)/(2*w)
    a1 = k+2
    a2 = w*w
    Hprime = 2*mp.mpf(19)*k/5-mp.mpf(261)/10
    transversal = -Hprime/(2*(a1*a1+a2))
    residuals = [mp.norm(J*q-mp.j*w*q), mp.norm(J.T*p+mp.j*w*p), abs(inner(p, q)-1)]
    assert max(residuals) < mp.mpf(10)**(-dps+8)
    return {"kappa": k, "omega": w, "q": q, "p": p, "G21": G21,
            "l1_qfirst1": l1, "l1_qunit": l1/inner(q, q),
            "real_eigenvalue_derivative": transversal, "max_residual": max(residuals)}


def exact_lyapunov_sign():
    """Calculate Re(G21) in Q[t]/(10*t^4+361*t^2+1935), t=i*omega.

    This field contains both Hopf thresholds under its two positive-imaginary
    embeddings. Complex conjugation is t -> -t. All operations are exact.
    """
    t = sp.symbols("t")
    field = sp.QQ.alg_field_from_poly(sp.Poly(10*t**4+361*t**2+1935, t), alias="t")
    one = field.one
    gen = field([1, 0])
    k = -(5*gen**2+25)/19

    def conj(a):
        coefficients = a.to_list()
        n = len(coefficients)-1
        return field([coefficient*(-1)**(n-i) for i, coefficient in enumerate(coefficients)])

    def solve(matrix, rhs):
        rows = [[field.convert(v) for v in row]+[field.convert(rhs[i])] for i, row in enumerate(matrix)]
        size = len(rows)
        for col in range(size):
            pivot = next(i for i in range(col, size) if rows[i][col])
            rows[col], rows[pivot] = rows[pivot], rows[col]
            scale = rows[col][col]
            rows[col] = [v/scale for v in rows[col]]
            for i in range(size):
                if i != col:
                    multiplier = rows[i][col]
                    rows[i] = [v-multiplier*w for v, w in zip(rows[i], rows[col])]
        return [row[-1] for row in rows]

    def B(u, v):
        return [
            -2*u[0]*v[0]-(u[0]*v[1]+u[1]*v[0])-10*(u[0]*v[2]+u[2]*v[0]),
            4*(u[0]*v[1]+u[1]*v[0])-2*u[1]*v[1]-(u[1]*v[2]+u[2]*v[1]),
            -k*(u[0]+8*u[1])*(v[0]+8*v[1])/450,
        ]

    J = [[-one, -one, -10*one], [4*one, -one, -one], [k/10, 4*k/5, -k]]
    qy = (41+gen)/(9+10*gen)
    q = [one, qy, 4-(1+gen)*qy]
    qb = [conj(v) for v in q]
    ell_xy = solve([[-one-gen, 4*one], [-one, -one-gen]], [-k/10, -4*k/5])
    ell = ell_xy+[one]
    normalizer = sum((u*v for u, v in zip(ell, q)), field.zero)
    ell = [v/normalizer for v in ell]
    assert all(sum((J[i][j]*q[j] for j in range(3)), field.zero) == gen*q[i] for i in range(3))
    assert all(sum((ell[i]*J[i][j] for i in range(3)), field.zero) == gen*ell[j] for j in range(3))
    assert sum((ell[i]*q[i] for i in range(3)), field.zero) == one
    h11 = solve(J, B(q, qb))
    resolvent = [[2*gen*(i == j)-J[i][j] for j in range(3)] for i in range(3)]
    h20 = solve(resolvent, B(q, q))
    cubic = [field.zero, field.zero, k*(q[0]+8*q[1])**2*(qb[0]+8*qb[1])/13500]
    b11 = B(q, h11)
    b20 = B(qb, h20)
    G = sum((ell[i]*(cubic[i]-2*b11[i]+b20[i]) for i in range(3)), field.zero)
    realG = (G+conj(G))/2
    coefficients = realG.to_list()
    polynomial = sum(sp.Rational(coefficient)*t**(len(coefficients)-1-i) for i, coefficient in enumerate(coefficients))
    assert sp.Poly(polynomial, t).degree() <= 2
    assert sp.expand(polynomial).coeff(t, 1) == 0
    kr = sp.symbols("kappa")
    expression = sp.factor(polynomial.subs(t**2, -5-sp.Rational(19, 5)*kr))
    # Both kappa thresholds lie between 0 and 7, as follows from
    # 0 < sqrt(52921) < 261 and sqrt(52921) < 271.
    assert expression.subs(kr, 0) < 0
    assert expression.subs(kr, 7) < 0
    assert sp.Poly(expression, kr).degree() <= 1
    print("EXACT Re(G21), q[0]=1:", expression)
    print("Exact sign certificate: affine expression negative at kappa=0 and 7; both thresholds in (0,7).")
    return expression


if __name__ == "__main__":
    exact_checks()
    exact_lyapunov_sign()
    for branch in [-1, 1]:
        results60 = evaluate(60, branch)
        results100 = evaluate(100, branch)
        discrepancy = abs(results60["l1_qfirst1"]-results100["l1_qfirst1"])
        assert discrepancy < mp.mpf("1e-50")
        print("\nBRANCH", "lower" if branch == -1 else "upper")
        for name, value in results100.items():
            print(name, "=", mp.nstr(value, 55))
        print("60/100 digit l1 discrepancy =", mp.nstr(discrepancy, 5))
