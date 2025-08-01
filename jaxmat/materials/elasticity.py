import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import IsotropicTensor4
from .behavior import SmallStrainBehavior


class LinearElasticIsotropic(eqx.Module):
    E: float = eqx.field(converter=jnp.asarray)
    nu: float = eqx.field(converter=jnp.asarray)
    internal = None

    @property
    def kappa(self):
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        return self.E / (2 * (1 + self.nu))

    @property
    def C(self):
        return IsotropicTensor4(self.kappa, self.mu)

    @property
    def S(self):
        return self.C.inv


class ElasticBehavior(SmallStrainBehavior):
    elasticity: eqx.Module
    internal = None

    def constitutive_update(self, eps, state, dt):
        sig = self.elasticity.C @ eps
        new_state = state.update(strain=eps, stress=sig)
        return sig, new_state
