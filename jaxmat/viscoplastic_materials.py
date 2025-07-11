from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from dolfinx_materials.jax_materials.tensors import dev, to_mat
from tensor_utils import jacobi_eig_3x3, eig33


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
        sI = eig33(to_mat(sig))[0]
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
