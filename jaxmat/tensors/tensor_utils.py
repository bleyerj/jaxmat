from functools import partial
import jax
import jax.numpy as jnp
from .linear_algebra import _sqrtm, eig33
from . import Tensor2, SymmetricTensor2, SymmetricTensor4


# These functions apply to tensors
@partial(jax.jit, static_argnums=1)
def polar(F, mode="RU"):
    """Computes the 'RU' or 'VR' polar decomposition of F."""
    C = F.T @ F
    U, U_inv = _sqrtm(jnp.asarray(C))
    R = Tensor2(F @ U_inv)
    if mode == "RU":
        return R, SymmetricTensor2(U)
    elif mode == "VR":
        V = (R @ U @ R.T).sym
        return V, R


def sym(A):
    """Computes the symmetric part of a tensor."""
    return 0.5 * (A + A.T)


def skew(A):
    """Computes the skew part of a tensor."""
    return 0.5 * (A - A.T)


def tr(A):
    """Trace of a n-dim 2nd-rank tensor."""
    return jnp.trace(A)


def dev(A):
    Id = SymmetricTensor2.identity()
    return A - Id * (tr(A) / A.dim)


def dev_vect(A):
    K = SymmetricTensor4.K().array
    return K @ A


def stretch_tensor(F):
    """Computes the strech tensor U = sqrtm(F.T @ F)."""
    return polar(F)[1]


@jax.custom_jvp
def eigenvalues(sig):
    eigvals, eigendyads = eig33(sig)
    return eigvals


@eigenvalues.defjvp
def eigenvalues_jvp(primals, tangents):
    (sig,) = primals
    (dsig,) = tangents
    eigvals, eigendyads = eig33(sig)
    deig = jnp.tensordot(eigendyads, dsig)
    return eigvals, deig


def to_mat(x):
    # FIXME: should be removed when working with Tensors
    if len(x) == 6:
        return jnp.array(
            [
                [x[0], x[3] / jnp.sqrt(2), x[4] / jnp.sqrt(2)],
                [x[3] / jnp.sqrt(2), x[1], x[5] / jnp.sqrt(2)],
                [x[4] / jnp.sqrt(2), x[5] / jnp.sqrt(2), x[2]],
            ]
        )
    else:
        return jnp.array(
            [
                [x[0], x[3], x[5]],
                [x[4], x[1], x[7]],
                [x[6], x[8], x[2]],
            ]
        )


def to_vect(X, symmetry=False):
    # FIXME: should be removed when working with Tensors
    if symmetry:
        return jnp.array(
            [
                X[0, 0],
                X[1, 1],
                X[2, 2],
                jnp.sqrt(2) * X[0, 1],
                jnp.sqrt(2) * X[0, 2],
                jnp.sqrt(2) * X[1, 2],
            ]
        )
    else:
        return jnp.array(
            [
                X[0, 0],
                X[1, 1],
                X[2, 2],
                X[0, 1],
                X[1, 0],
                X[0, 2],
                X[2, 0],
                X[1, 2],
                X[2, 1],
            ]
        )
