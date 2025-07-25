from functools import partial
import jax
from .linear_algebra import _sqrtm
from . import SymmetricTensor2


@partial(jax.jit, static_argnums=1)
def polar(F, mode="RU"):
    """Computes the 'RU' or 'VR' polar decomposition of F."""
    C = (F.T @ F).sym
    U, U_inv = _sqrtm(C)
    R = F @ U_inv
    if mode == "RU":
        return R, SymmetricTensor2(U)
    elif mode == "VR":
        V = (R @ U @ R.T).sym
        return V, R


def stretch_tensor(F):
    """Computes the strech tensor U = sqrtm(F.T @ F)."""
    return polar(F)[1]
