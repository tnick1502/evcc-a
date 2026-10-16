"""Exact convex-hull certificate for two groups of four-property measurements.

If strictly positive weights give the same convex combination of both groups
and the points have affine rank four, no nonconstant linear projection of the
properties can place the groups in two separate spans of an order.

Usage: ``python -m evcca.certificate results/tolsa_hull_witnesses.json``
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction


def rank(matrix):
    """Rank of a matrix of Fractions (Gaussian elimination)."""
    a = [list(row) for row in matrix]
    r = 0
    for col in range(len(a[0])):
        pivot = next((i for i in range(r, len(a)) if a[i][col]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        v = a[r][col]
        a[r] = [z / v for z in a[r]]
        for i in range(r + 1, len(a)):
            q = a[i][col]
            if q:
                a[i] = [u - q * w for u, w in zip(a[i], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def _solve(a, b):
    """Solve a square linear system of Fractions."""
    a = [list(row) + [v] for row, v in zip(a, b)]
    n = len(a)
    for col in range(n):
        p = next(i for i in range(col, n) if a[i][col])
        a[col], a[p] = a[p], a[col]
        v = a[col][col]
        a[col] = [x / v for x in a[col]]
        for i in range(n):
            if i != col:
                q = a[i][col]
                if q:
                    a[i] = [x - q * y for x, y in zip(a[i], a[col])]
    return [row[-1] for row in a]


def construct(left, right, approximate):
    """Turn floating-point hull weights into exact rational weights.

    ``left`` and ``right`` are the measurements of the two groups (rows of
    four values); ``approximate`` are weights from a linear program.
    """
    import numpy as np
    from scipy.linalg import qr

    left = [[str(float(x)) for x in row] for row in left]
    right = [[str(float(x)) for x in row] for row in right]
    a = [[Fraction(x) for x in row] for row in left]
    b = [[Fraction(x) for x in row] for row in right]
    na, nb = len(a), len(b)
    m = [[row[j] for row in a] + [-row[j] for row in b] for j in range(4)]
    m += [[Fraction(1)] * na + [Fraction(0)] * nb, [Fraction(0)] * na + [Fraction(1)] * nb]
    numeric = np.array(m, dtype=float)
    numeric /= np.linalg.norm(numeric, axis=1)[:, None]
    pivots = qr(numeric, pivoting=True, mode="economic")[2][:6]
    w = [Fraction(str(float(x))) for x in approximate]
    target = [Fraction(0)] * 4 + [Fraction(1), Fraction(1)]
    residual = [t - sum(x * y for x, y in zip(row, w)) for row, t in zip(m, target)]
    correction = _solve([[row[int(i)] for i in pivots] for row in m], residual)
    for i, delta in zip(pivots, correction):
        w[int(i)] += delta
    result = dict(
        left_decimal_points=left,
        right_decimal_points=right,
        left_rational_weights=[str(x) for x in w[:na]],
        right_rational_weights=[str(x) for x in w[na:]],
        input_interpretation="Exact decimal representation of the measurements.",
    )
    verify(result)
    return result


def verify(certificate):
    """Check positivity, unit sums, the four equalities and affine rank four."""
    a = [[Fraction(x) for x in row] for row in certificate["left_decimal_points"]]
    b = [[Fraction(x) for x in row] for row in certificate["right_decimal_points"]]
    wa = [Fraction(x) for x in certificate["left_rational_weights"]]
    wb = [Fraction(x) for x in certificate["right_rational_weights"]]
    if not (min(wa + wb) > 0 and sum(wa) == sum(wb) == 1):
        raise ValueError("Weights are not strictly positive with unit sums.")
    for j in range(4):
        if sum(w * x[j] for w, x in zip(wa, a)) != sum(w * x[j] for w, x in zip(wb, b)):
            raise ValueError(f"Convex combinations differ in property {j}.")
    points = a + b
    if rank([[x[j] - points[0][j] for j in range(4)] for x in points[1:]]) != 4:
        raise ValueError("Points do not have affine rank four.")
    return True


def main(argv=None):
    path = (argv or sys.argv[1:])[0]
    witnesses = json.load(open(path))["witnesses"]
    certificates = [w["exact_decimal_certificate"] for w in witnesses if "exact_decimal_certificate" in w]
    if not certificates:
        raise SystemExit("No exact certificate in " + path)
    for certificate in certificates:
        verify(certificate)
    print(f"Verified {len(certificates)} certificate(s): positive weights, equal combinations, affine rank 4.")


if __name__ == "__main__":
    main()
