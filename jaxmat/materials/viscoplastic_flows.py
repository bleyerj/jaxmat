from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.state import make_batched


class VoceHardening(eqx.Module):
    sig0: float
    sigu: float
    b: float

    def __call__(self, p):
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-1.0 * self.b * p))


class NortonFlow(eqx.Module):
    K: float
    m: float

    def __call__(self, f):
        return jnp.maximum(f / self.K, 0) ** self.m


class AbstractKinematicHardening(eqx.Module):
    nvars: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, X, *args):
        pass

    def sig_eff(self, sig, X):
        return sig - jnp.sum(X, axis=0)


class ArmstrongFrederickHardening(AbstractKinematicHardening):
    C: jax.Array
    g: jax.Array
    nvars = 2

    def __call__(self, a, dp, depsp):
        return make_batched(depsp, self.nvars) - self.g * a * dp

    def sig_eff(self, sig, a):
        return sig - 2 / 3 * self.C * jnp.sum(a, axis=0)
