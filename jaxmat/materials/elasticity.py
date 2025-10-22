import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import IsotropicTensor4
from .behavior import SmallStrainBehavior


class LinearElasticIsotropic(eqx.Module):
    """An isotropic linear elastic model."""

    E: float = eqx.field(converter=jnp.asarray)
    r"""Young modulus $E$"""
    nu: float = eqx.field(converter=jnp.asarray)
    r"""Poisson ratio $\nu$"""

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

    @property
    def C(self):
        r"""4th-rank isotropic stiffness tensor

        $$\mathbb{C}=3\kappa\mathbb{J}+2\mu\mathbb{K}$$

        where $\mathbb{J}$ and $\mathbb{K}$ are the hydrostatic and deviatoric projectors.
        """
        return IsotropicTensor4(self.kappa, self.mu)

    @property
    def S(self):
        r"""4th-rank isotropic compliance tensor:

        $$\mathbb{S}=\mathbb{C}^{-1} = \dfrac{1}{3\kappa}\mathbb{J}+\dfrac{1}{2\mu}\mathbb{K}$$
        """
        return self.C.inv

    def strain_energy(self, eps):
        """Strain energy density

        $$\psi(\beps)=\dfrac{1}{2}\beps:\mathbb{C}:\beps$$

        Parameters
        ----------
        eps: SymmetricTensor2
            Strain tensor
        """
        return 0.5 * jnp.trace(eps @ (self.C @ eps))


class ElasticBehavior(SmallStrainBehavior):
    """A small strain linear elastic behavior."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""

    def constitutive_update(self, eps, state, dt):
        sig = self.elasticity.C @ eps
        new_state = state.update(strain=eps, stress=sig)
        return sig, new_state
