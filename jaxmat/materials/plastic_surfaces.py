from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import eigenvalues, dev


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
