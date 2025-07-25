from functools import partial
import jax
import jax.numpy as jnp
from .linear_algebra import _sqrtm
from . import Tensor2, SymmetricTensor2


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
    return jnp.trace(A)


def dev(A):
    print(A)
    Id = SymmetricTensor2.identity()
    return A - tr(A) / A.dim * Id


def stretch_tensor(F):
    """Computes the strech tensor U = sqrtm(F.T @ F)."""
    return polar(F)[1]
