from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import SymmetricTensor4, to_mat, eigenvalues, to_vect, dev

# from jaxmat.tensors import dev_vect as dev

# K = SymmetricTensor4.K().array
# J = SymmetricTensor4.J().array


# class LinearElasticIsotropic(eqx.Module):
#     E: float = eqx.field(converter=jax.numpy.asarray)
#     nu: float = eqx.field(converter=jax.numpy.asarray)

#     @property
#     def kappa(self):
#         return self.E / (3 * (1 - 2 * self.nu))

#     @property
#     def mu(self):
#         return self.E / (2 * (1 + self.nu))

#     @property
#     def C(self):
#         return 3 * self.kappa * J + 2 * self.mu * K

#     @property
#     def S(self):
#         return 1 / (3 * self.kappa) * J + 1 / (2 * self.mu) * K

#     def constitutive_update(self, eps, state, dt):
#         sig = to_mat(self.C @ to_vect(eps, True))
#         state = state.update(strain=eps, stress=sig)
#         return sig, state


class AbstractPlasticSurface(eqx.Module):
    tol: float = 1e-8

    @abstractmethod
    def __call__(self, sig):
        pass

    def normal(self, sig):
        return jax.jacfwd(lambda sig: jnp.clip(self.__call__(sig), a_min=self.tol))(sig)


class vonMises(AbstractPlasticSurface):
    def __call__(self, sig):
        return jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig))


class Hosford(AbstractPlasticSurface):
    a: float = 2.0

    def __call__(self, sig):
        sI = eigenvalues(sig)
        return (
            1
            / 2
            * (
                jnp.abs(sI[0] - sI[1]) ** self.a
                + jnp.abs(sI[0] - sI[2]) ** self.a
                + jnp.abs(sI[2] - sI[1]) ** self.a
            )
        ) ** (1 / self.a)


class VoceHardening(eqx.Module):
    sig0: float = eqx.field(converter=jax.numpy.asarray)
    sigu: float = eqx.field(converter=jax.numpy.asarray)
    b: float = eqx.field(converter=jax.numpy.asarray)

    def __call__(self, p):
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-self.b * p))


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
