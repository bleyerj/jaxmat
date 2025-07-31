import jax
import jax.numpy as jnp
import equinox as eqx


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


class ArmstrongFrederickHardening(eqx.Module):
    C: jax.Array
    g: jax.Array

    def sig_eff(self, sig, a):
        return sig - 2 / 3 * self.C * jnp.sum(a, axis=0)
