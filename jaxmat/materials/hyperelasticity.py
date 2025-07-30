import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors.linear_algebra import invariants_principal
from jaxmat.tensors.linear_algebra import det33 as det


class HyperelasticPotential(eqx.Module):
    def PK1(self, F):
        return jax.jacfwd(self.__call__)(F)

    def PK2(self, F):
        return (F.inv @ self.PK1(F)).sym

    def Cauchy(self, F):
        J = det(F)
        # Divide on the right rather than on the left to preserve Tensor object due to operator dispatch priority.
        return (self.PK1(F) @ F.T / J).sym


class Hyperelasticity(eqx.Module):
    potential: HyperelasticPotential

    def constitutive_update(self, F, state, dt):
        PK1 = self.potential.PK1(F)
        PK2 = self.potential.PK2(F)
        sig = self.potential.Cauchy(F)
        new_state = state.update(stress=PK1, PK2=PK2, Cauchy=sig)
        return PK1, new_state


class VolumetricPart(eqx.Module):
    beta: float = 2.0

    def __call__(self, J):
        return 1 / self.beta**2 * (J ** (self.beta) - self.beta * jnp.log(J) - 1)


class CompressibleNeoHookean(HyperelasticPotential):
    mu: float
    kappa: float
    volumetric: eqx.Module = VolumetricPart()

    def __call__(self, F):
        C = F.T @ F
        I1, I2, I3 = invariants_principal(C)
        J = jnp.sqrt(I3)
        return self.mu / 2 * (I1 - 3 - 2 * jnp.log(J)) + self.kappa * self.volumetric(J)


class CompressibleGhentMooneyRivlin(HyperelasticPotential):
    c1: float
    c2: float
    Jm: float
    kappa: float
    volumetric: eqx.Module = VolumetricPart()

    def __call__(self, F):
        C = F.T @ F
        I1, I2, I3 = invariants_principal(C)
        J = jnp.sqrt(I3)
        arg = 1 - (I1 - 3 - 2 * jnp.log(J)) / self.Jm
        return (
            -0.5 * self.c1 * self.Jm * jnp.log(arg)
            + 0.5 * self.c2 * (I2 - 3)
            + self.volumetric(J)
        )
