from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import eigenvalues, dev, SymmetricTensor2
from jaxmat.tensors.utils import safe_norm


def safe_zero(method):
    """Decorator for yield surfaces to avoid NaNs for zero stress in both fwd and bwd AD."""

    def wrapper(self, x):
        x_norm = jnp.linalg.norm(x)
        x_safe = SymmetricTensor2(tensor=jnp.where(x_norm > 0, x, x))
        return jnp.where(x_norm > 0, method(self, x_safe), 0.0)

    return wrapper


class AbstractPlasticSurface(eqx.Module):
    @abstractmethod
    def __call__(self, sig):
        pass

    def normal(self, sig):
        return jax.jacfwd(self.__call__)(sig)


class vonMises(AbstractPlasticSurface):

    @safe_zero
    def __call__(self, sig):
        return jnp.sqrt(3 / 2.0) * safe_norm(dev(sig))


class Hosford(AbstractPlasticSurface):
    a: float = eqx.field(converter=jnp.asarray, default=2.0)

    @safe_zero
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
