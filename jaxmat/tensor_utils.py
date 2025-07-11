import jax
import jax.numpy as jnp


def inv33(A):
    """Explicit inverse of a 3x3 matrix A using cofactor formula."""
    # Minors and cofactors
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]

    # Cofactor matrix (transposed for adjugate directly)
    cof = jnp.array(
        [
            [a22 * a33 - a23 * a32, a13 * a32 - a12 * a33, a12 * a23 - a13 * a22],
            [a23 * a31 - a21 * a33, a11 * a33 - a13 * a31, a13 * a21 - a11 * a23],
            [a21 * a32 - a22 * a31, a12 * a31 - a11 * a32, a11 * a22 - a12 * a21],
        ]
    )

    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )

    invA = cof.T / det
    return invA


def invariants_principal(A):
    """Principal invariants of a real 3×3 tensor A."""
    i1 = jnp.trace(A)
    i2 = (jnp.trace(A) ** 2 - jnp.trace(A.dot(A))) / 2
    i3 = jnp.linalg.det(A)
    return i1, i2, i3


def invariants_main(A):
    """J-invariants: trace(A), trace(A^2), trace(A^3)."""
    j1 = jnp.trace(A)
    j2 = jnp.trace(A.dot(A))
    j3 = jnp.trace(A.dot(A).dot(A))
    return j1, j2, j3


@jax.jit
def eig33(A):
    """Return sorted eigenvalues λ0 ≤ λ1 ≤ λ2 and eigenprojectors E0, E1, E2."""

    def eigvals(A, eps=3e-16):
        q = jnp.trace(A) / 3.0

        # B = A - q I
        B = A - q * jnp.eye(3)

        # j = tr(B^2)/2
        j = jnp.trace(B @ B) / 2.0

        # b = tr(B^3)/3
        b = jnp.trace(B @ B @ B) / 3.0

        # p = 2/sqrt(3) * sqrt(j + eps^2)
        p = 2.0 / jnp.sqrt(3.0) * jnp.sqrt(j + eps**2)

        # r = 4 b / p^3
        r = 4.0 * b / (p**3 + eps)  # add eps in denominator to avoid div by zero

        # clip r to [-1 + eps, 1 - eps]
        r = jnp.clip(r, -1.0 + eps, 1.0 - eps)

        # phi = acos(r) / 3
        phi = jnp.arccos(r) / 3.0

        # eigenvalues sorted λ0 <= λ1 <= λ2
        λ0 = q + p * jnp.cos(phi + 2.0 * jnp.pi / 3.0)
        λ1 = q + p * jnp.cos(phi + 4.0 * jnp.pi / 3.0)
        λ2 = q + p * jnp.cos(phi)
        lam_arr = jnp.asarray([λ0, λ1, λ2])
        return lam_arr, lam_arr

    E, λ = jax.jacfwd(eigvals, has_aux=True)(A)

    return λ, E


import jax
import jax.numpy as jnp
from functools import partial

# @jax.jit
import jax
import jax.numpy as jnp
from jax import lax

import jax
import jax.numpy as jnp
from jax import lax


@jax.jit
def jacobi_eig_3x3(A, max_iter=50, tol=1e-6):
    """
    JAX-compatible Jacobi diagonalization for a symmetric 3x3 matrix.

    Args:
        A: (3, 3) symmetric matrix
        max_iter: maximum number of sweeps
        tol: convergence threshold for off-diagonal sum

    Returns:
        W: (3,) eigenvalues
        Q: (3, 3) eigenvectors (columns)
    """
    Q = jnp.eye(3)
    W = jnp.diag(A)

    def off_diag_sum(A):
        return jnp.sum(jnp.abs(jnp.array([A[0, 1], A[0, 2], A[1, 2]])))

    def sweep(state, _):
        A, Q, W = state

        def apply_rotation(pair, state):
            A, Q, W = state
            x, y = pair
            axy = A[x, y]
            abs_axy = jnp.abs(axy)

            def rotated(operand):
                A, Q, W, x, y, axy = operand
                h = W[y] - W[x]
                theta = 0.5 * h / axy
                t = jnp.where(
                    h == 0.0,
                    1.0 / jnp.abs(axy),
                    jnp.sign(theta) / (jnp.abs(theta) + jnp.sqrt(1 + theta**2)),
                )
                c = 1.0 / jnp.sqrt(1 + t**2)
                s = t * c
                z = t * axy

                W = W.at[x].add(-z)
                W = W.at[y].add(z)

                A = A.at[x, y].set(0.0)
                A = A.at[y, x].set(0.0)

                def update_A(r, A):
                    cond = jnp.logical_and(r != x, r != y)
                    i1, j1 = jnp.minimum(r, x), jnp.maximum(r, x)
                    i2, j2 = jnp.minimum(r, y), jnp.maximum(r, y)
                    Arx = A[i1, j1]
                    Ary = A[i2, j2]
                    A1 = A.at[i1, j1].set(c * Arx - s * Ary)
                    A1 = A1.at[i2, j2].set(s * Arx + c * Ary)
                    return lax.select(cond, A1, A)

                A = lax.fori_loop(0, 3, update_A, A)

                def update_Q(r, Q):
                    Qrx = Q[r, x]
                    Qry = Q[r, y]
                    Q = Q.at[r, x].set(c * Qrx - s * Qry)
                    Q = Q.at[r, y].set(s * Qrx + c * Qry)
                    return Q

                Q = lax.fori_loop(0, 3, update_Q, Q)

                return A, Q, W

            operand = (A, Q, W, x, y, axy)
            A, Q, W = lax.cond(abs_axy < tol, lambda _: (A, Q, W), rotated, operand)
            return A, Q, W

        pairs = jnp.array([[0, 1], [0, 2], [1, 2]])
        for i in range(3):
            A, Q, W = apply_rotation(pairs[i], (A, Q, W))

        return (A, Q, W), None

    def cond_fn(state):
        A, _, _ = state
        return off_diag_sum(A) > tol

    def body_fn(state):
        (A, Q, W), _ = sweep(state, None)
        return A, Q, W

    def loop_body(i, state):
        return lax.cond(cond_fn(state), body_fn, lambda s: s, state)

    state = (A, Q, W)
    state = lax.fori_loop(0, max_iter, loop_body, state)
    _, Q, W = state
    return W, Q


# A = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
# print(jacobi_eig_3x3(A)[0])
