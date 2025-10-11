from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from .utils import safe_norm, safe_sqrt


def dim(A):
    """Dimension of a n-rank tensor, assuming shape=(dim, dim, ..., dim)."""
    return A.shape[0]


def tr(A):
    """Trace of a n-dim 2nd-rank tensor."""
    return jnp.trace(A)


def dev(A):
    """Deviatoric part of a n-dim 2nd-rank tensor."""
    d = dim(A)
    Id = jnp.eye(d)
    return A - tr(A) / d * Id


def det33(A):
    a11, a12, a13 = A[0, 0], A[0, 1], A[0, 2]
    a21, a22, a23 = A[1, 0], A[1, 1], A[1, 2]
    a31, a32, a33 = A[2, 0], A[2, 1], A[2, 2]
    return (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )


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

    invA = cof / det
    return invA


def invariants_principal(A):
    """Principal invariants of a real 3x3 tensor A."""
    i1 = jnp.trace(A)
    i2 = (jnp.trace(A) ** 2 - jnp.trace(A @ A)) / 2
    i3 = det33(A)
    return i1, i2, i3


def invariants_main(A):
    """J-invariants: trace(A), trace(A^2), trace(A^3)."""
    j1 = jnp.trace(A)
    j2 = jnp.trace(A.dot(A))
    j3 = jnp.trace(A.dot(A).dot(A))
    return j1, j2, j3


def pq_invariants(sig):
    r"""Hydrostatic/deviatoric equivalent stresses $(p,q)$. Typically used in soil mechanics.

    $$p = - \tr(\bsig)/3 = -I_1/3$$
    $$q = \sqrt{\frac{3}{2}\bs:\bs} = \sqrt{3 J_2}$$
    """
    p = -jnp.trace(sig) / 3
    s = dev(sig)
    q = safe_sqrt(3.0 / 2.0 * jnp.vdot(s, s))
    return p, q


@partial(jax.jit, static_argnums=1)
def eig33(A, rtol=1e-16):
    # def dyad_3_distinct(A, lamb):
    #     """
    #     Hartmann, S. (2019) “Computational Aspects of the Symmetric Eigenvalue Problem of Second Order Tensors”,
    #     Technische Mechanik - European Journal of Engineering Mechanics, 23(2-4), pp. 283–294.
    #     Available at: https://journals.ub.ovgu.de/index.php/techmech/article/view/989
    #     """
    #     Id = jnp.eye(3)
    #     A2 = A @ A
    #     d = jnp.array([lamb[0] - lamb[1], lamb[1] - lamb[2], lamb[2] - lamb[1]])
    #     D = jnp.array([-d[0] * d[2], -d[0] * d[1], -d[1] * d[2]])
    #     h1 = jnp.array([lamb[1] * lamb[2], -(lamb[1] + lamb[2]), 1]) / D[0]
    #     N1 = h1[0] * Id + h1[1] * A + h1[2] * A2
    #     h2 = jnp.array([lamb[0] * lamb[2], -(lamb[0] + lamb[2]), 1]) / D[1]
    #     N2 = h2[0] * Id + h2[1] * A + h2[2] * A2
    #     h3 = jnp.array([lamb[0] * lamb[1], -(lamb[0] + lamb[1]), 1]) / D[2]
    #     N3 = h3[0] * Id + h3[1] * A + h3[2] * A2
    #     return (N1, N2, N3)

    def compute_eigvals_HarariAlbocher(A):
        """
        Eigendecomposition of 3x3 symmetric matrix based on
        Harari, I., & Albocher, U. (2023). Computation of eigenvalues of a real,
        symmetric 3× 3 matrix with particular reference to the pernicious case of two nearly equal eigenvalues.
        International Journal for Numerical Methods in Engineering, 124(5), 1089-1110.
        """
        A = jnp.asarray(A)
        norm = safe_norm(A)
        Id = jnp.eye(dim(A))
        I1 = jnp.trace(A)
        S = dev(A)
        J2 = tr(S.T @ S) / 2
        s = safe_sqrt(J2 / 3)

        def branch_near_iso(_):
            eigvals = jnp.ones((3,)) * I1 / 3
            return eigvals, eigvals

        def branch_general(_):
            T = S @ S - 2 * J2 / 3 * Id
            d = safe_norm(T - s * S) / safe_norm(T + s * S)
            sj = jnp.sign(1 - d)
            cond = sj * (1 - d) < rtol * norm

            def branch_two_eigvals(_):
                lamb_max = jnp.sqrt(3) * s
                eigvals_dev = jnp.array([lamb_max, 0.0, -lamb_max])
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            def branch_three_eigvals(_):
                alpha = 2 / 3 * jnp.arctan(d**sj)
                lambda_d = 2 * sj * s * jnp.cos(alpha)
                sd = jnp.sqrt(3) * s * jnp.sin(alpha)

                eigvals_dev = lax.cond(
                    lambda_d > 0,
                    lambda _: jnp.array(
                        [-lambda_d / 2 - sd, -lambda_d / 2 + sd, lambda_d]
                    ),
                    lambda _: jnp.array(
                        [lambda_d, -lambda_d / 2 - sd, -lambda_d / 2 + sd]
                    ),
                    operand=None,
                )
                eigvals = eigvals_dev + I1 / 3
                return eigvals, eigvals

            return lax.cond(
                cond, branch_two_eigvals, branch_three_eigvals, operand=None
            )

        return lax.cond(s < rtol * norm, branch_near_iso, branch_general, operand=None)

    eigendyads, eigvals = jax.jacfwd(compute_eigvals_HarariAlbocher, has_aux=True)(A)

    return eigvals, eigendyads


def _sqrtm(C):
    """
    Unified expression for sqrt and inverse sqrt of a symmetric matrix $\bC$,
    see Simo & Hugues, Computational Inelasticity, p.244
    """
    Id = jnp.eye(3)
    C2 = C @ C
    eigvals, _ = eig33(C)
    lamb = safe_sqrt(eigvals)
    i1 = jnp.sum(lamb)
    i2 = lamb[0] * lamb[1] + lamb[1] * lamb[2] + lamb[0] * lamb[2]
    i3 = jnp.prod(lamb)
    D = i1 * i2 - i3
    U = 1 / D * (-C2 + (i1**2 - i2) * C + i1 * i3 * Id)
    U_inv = 1 / i3 * (C - i1 * U + i2 * Id)
    return U, U_inv


def sqrtm(A):
    """Matrix square-root of a symmetric 3x3 matrix."""
    return _sqrtm(A)[0]


def inv_sqrtm(A):
    """Matrix inverse square-root of a symmetric 3x3 matrix."""
    return _sqrtm(A)[1]


def isotropic_function(fun, A):
    r"""Computes an isotropic function of a symmetric 3x3 matrix.

    Parameters
    ----------
    fun : callable
        A scalar function f(x)
    A : jax.Array
        A symmetric 3x3 matrix

    Returns
    -------
    jax.Array
        A new 3x3 matrix such that `f_A = sum_i f(\lambda_i) n_i \times n_i`
    """
    eigvals, eigendyads = eig33(A)
    f = fun(eigvals)
    return sum([fi * Ni for fi, Ni in zip(f, eigendyads)])


def expm(A):
    """Matrix exponential of a symmetric 3x3 matrix."""
    return isotropic_function(jnp.exp, A)


def logm(A):
    """Matrix logarithm of a symmetric 3x3 matrix."""
    return isotropic_function(jnp.log, A)


def powm(A, m):
    """Matrix power of exponent m of a symmetric 3x3 matrix."""
    return isotropic_function(lambda x: jnp.power(x, m), A)
