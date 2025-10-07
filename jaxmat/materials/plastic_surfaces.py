from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import eigenvalues, dev, SymmetricTensor2
from jaxmat.tensors.utils import safe_norm


def safe_zero(method):
    """Decorator for yield surfaces to avoid NaNs for zero stress in both fwd and bwd AD."""

    def wrapper(self, x, *args):
        x_norm = jnp.linalg.norm(x)
        x_safe = SymmetricTensor2(tensor=jnp.where(x_norm > 0, x, x))
        return jnp.where(x_norm > 0, method(self, x_safe, *args), 0.0)

    return wrapper


class AbstractPlasticSurface(eqx.Module):
    """Abstract plastic surface class."""

    @abstractmethod
    def __call__(self, sig, *args):
        """Yield surface expression.

        Note: We recommend using the `safe_zero` decorator on this method to avoid
        NaNs for zero stresses.

        Parameters
        ----------
        sig: Tensor
            Stress tensor.
        args: tuple
            Additional thermodynamic forces entering the yield surface definition.
        """
        pass

    def normal(self, sig, *args):
        """Normal to the yield surface. Computed automatically using forward AD on `__call__`.

        Parameters
        ----------
        sig: Tensor
            Stress tensor.
        args: tuple
            Additional thermodynamic forces entering the yield surface definition.
        """
        return jax.jacfwd(self.__call__, argnums=0)(sig, *args)


class vonMises(AbstractPlasticSurface):
    r"""von Mises yield surface

    $$\sqrt{\dfrac{3}{2}\bs:\bs}$$

    where $\bs = \dev(\bsig)$"""

    @safe_zero
    def __call__(self, sig):
        return jnp.sqrt(3 / 2.0) * safe_norm(dev(sig))


class Hosford(AbstractPlasticSurface):
    r"""Hosford yield surface

    $$\left(\dfrac{1}{2}(|\sigma_\text{I}-\sigma_\text{II}|^a +
    |\sigma_\text{II}-\sigma_\text{III}|^a +
    |\sigma_\text{I}-\sigma_\text{III}|^a)\right)^{1/a}$$

    with $\sigma_\text{J}$ being the stress principal values.

    Parameters
    ----------
    a : float
        Hosford shape parameter
    """

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
