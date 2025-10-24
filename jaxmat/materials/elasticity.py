from abc import abstractmethod
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import IsotropicTensor4
from jaxmat.utils import enforce_dtype
from .behavior import SmallStrainBehavior


class AbstractLinearElastic(eqx.Module):
    """Small-strain elastic model."""

    @property
    @abstractmethod
    def C(self):
        pass

    @property
    def S(self):
        r"""4th-rank isotropic compliance tensor
        $$\mathbb{S}=\mathbb{C}^{-1}$$
        """
        return self.C.inv

    def strain_energy(self, eps):
        r"""Strain energy density

        $$\psi(\beps)=\dfrac{1}{2}\beps:\mathbb{C}:\beps$$

        Parameters
        ----------
        eps: SymmetricTensor2
            Strain tensor
        """
        return 0.5 * jnp.trace(eps @ (self.C @ eps))


class LinearElasticIsotropic(AbstractLinearElastic):
    """An isotropic linear elastic model."""

    E: float = enforce_dtype()
    r"""Young modulus $E$"""
    nu: float = enforce_dtype()
    r"""Poisson ratio $\nu$"""

    @property
    def C(self):
        r"""4th-rank isotropic stiffness tensor

        $$\mathbb{C}=3\kappa\mathbb{J}+2\mu\mathbb{K}$$

        where $\mathbb{J}$ and $\mathbb{K}$ are the hydrostatic and deviatoric projectors.
        """
        return IsotropicTensor4(self.kappa, self.mu)

    @property
    def kappa(self):
        r"""
        Bulk modulus

        $$\kappa = \dfrac{E}{3(1-2\nu)} = \lambda +\frac{2}{3}\mu$$
        """
        return self.E / (3 * (1 - 2 * self.nu))

    @property
    def mu(self):
        r"""
        Shear modulus

        $$\mu = \dfrac{E}{2(1+\nu)}$$
        """
        return self.E / (2 * (1 + self.nu))

    @property
    def lmbda(self):
        r"""
        Lamé modulus

        $$\lambda = \dfrac{E\nu}{(1+\nu)(1-2\nu)}$$
        """
        return self.E * self.nu / (1 + self.nu) / (1 - 2 * self.nu)


class ElasticBehavior(SmallStrainBehavior):
    """A small strain linear elastic behavior."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""
    internal = None

    def constitutive_update(self, eps, state, dt):
        sig = self.elasticity.C @ eps
        new_state = state.update(strain=eps, stress=sig)
        return sig, new_state
