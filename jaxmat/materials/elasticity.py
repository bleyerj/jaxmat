from abc import abstractmethod
import jax.numpy as jnp
import jax.scipy as jsc
import equinox as eqx
from jaxmat.tensors import IsotropicTensor4, SymmetricTensor4
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
    
    
class LinearElastic(AbstractLinearElastic):
    """A generic linear elastic model with custom stiffness tensor."""
    _C: SymmetricTensor4
    
    def __init__(self, C: SymmetricTensor4):
        """
        Initialize a linear elastic material with a given stiffness tensor.
        
        Args:
            C: 4th order stiffness tensor (SymmetricTensor4)
        """
        self._C = C
    
    @property
    def C(self) -> SymmetricTensor4:
        """Return the stiffness tensor."""
        return self._C


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
    
class LinearElasticOrthotropic(AbstractLinearElastic):
    """An isotropic linear elastic model."""

    EL: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{L}$"""
    ET: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{T}$"""
    EN: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{N}$"""
    nuLT: float = enforce_dtype()
    r"""Longitudinal-transverse Poisson coefficient $\nu_{LT}$"""
    nuLN: float = enforce_dtype()
    r"""Longitudinal-normal Poisson coefficient $\nu_{LN}$"""
    nuTN: float = enforce_dtype()
    r"""Transverse-normal Poisson coefficient $\nu_{TN}$"""
    muLT: float = enforce_dtype()
    r"""Longitudinal-transverse shear modulus $\mu_{LT}$"""
    muLN: float = enforce_dtype()
    r"""Longitudinal-normal shear modulus $\mu_{LN}$"""
    muTN: float = enforce_dtype()
    r"""Transverse-normal shear modulus $\mu_{TN}$"""

    @property
    def C(self):
        """Build stiffness matrix for orthotropic material.
            
        Returns
        -------
        ndarray 6x6 stiffness matrix in Voigt notation
            
        Notes
        -----
        Convention: L=x, T=y, N=z in Voigt notation [xx, yy, zz, yz, xz, xy]
        """
        print("Building orthotropic stiffness tensor:")
        print(f"Young's modulus (L): {self.EL}")
        print(f"Young's modulus (T): {self.ET}")
        print(f"Young's modulus (N): {self.EN}")
        print(f"Poisson ratio (nu_LT): {self.nuLT}")
        print(f"Poisson ratio (nu_LN): {self.nuLN}")
        print(f"Poisson ratio (nu_TN): {self.nuTN}")
        print(f"Shear modulus (mu_LT): {self.muLT}")
        print(f"Shear modulus (mu_LN): {self.muLN}")
        print(f"Shear modulus (mu_TN): {self.muTN}")
        # Build compliance matrix
        S_diag = jnp.array(
            [[1. / self.EL, -self.nuLT / self.EL, -self.nuLN / self.EL],
             [-self.nuLT / self.EL, 1 / self.ET, -self.nuTN / self.ET],
             [-self.nuLN / self.EL, -self.nuTN / self.ET, 1 / self.EN]])
        
        S_shear = jnp.diag(jnp.array(
            [1. / (2. * self.muTN), 
             1. / (2. * self.muLN), 
             1./(2. * self.muLT)]
            )
            )  # [yz, xz, xy] order
        S_Mandel = jsc.linalg.block_diag(S_diag, S_shear)
        C_Mandel = jnp.linalg.inv(S_Mandel)
        return SymmetricTensor4(array = C_Mandel)
    
class LinearElasticOrthotropic(AbstractLinearElastic):
    """An isotropic linear elastic model."""

    EL: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{L}$"""
    ET: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{T}$"""
    EN: float = enforce_dtype()
    r"""Longitudinal Young modulus $E_{N}$"""
    nuLT: float = enforce_dtype()
    r"""Longitudinal-transverse Poisson coefficient $\nu_{LT}$"""
    nuLN: float = enforce_dtype()
    r"""Longitudinal-normal Poisson coefficient $\nu_{LN}$"""
    nuTN: float = enforce_dtype()
    r"""Transverse-normal Poisson coefficient $\nu_{TN}$"""
    muLT: float = enforce_dtype()
    r"""Longitudinal-transverse shear modulus $\mu_{LT}$"""
    muLN: float = enforce_dtype()
    r"""Longitudinal-normal shear modulus $\mu_{LN}$"""
    muTN: float = enforce_dtype()
    r"""Transverse-normal shear modulus $\mu_{TN}$"""

    @property
    def C(self):
        """Build stiffness matrix for orthotropic material.
            
        Returns
        -------
        ndarray 6x6 stiffness matrix in Voigt notation
            
        Notes
        -----
        Convention: L=x, T=y, N=z in Voigt notation [xx, yy, zz, yz, xz, xy]
        """
        print("Building orthotropic stiffness tensor:")
        print(f"Young's modulus (L): {self.EL}")
        print(f"Young's modulus (T): {self.ET}")
        print(f"Young's modulus (N): {self.EN}")
        print(f"Poisson ratio (nu_LT): {self.nuLT}")
        print(f"Poisson ratio (nu_LN): {self.nuLN}")
        print(f"Poisson ratio (nu_TN): {self.nuTN}")
        print(f"Shear modulus (mu_LT): {self.muLT}")
        print(f"Shear modulus (mu_LN): {self.muLN}")
        print(f"Shear modulus (mu_TN): {self.muTN}")
        # Build compliance matrix
        S_diag = jnp.array(
            [[1. / self.EL, -self.nuLT / self.EL, -self.nuLN / self.EL],
             [-self.nuLT / self.EL, 1 / self.ET, -self.nuTN / self.ET],
             [-self.nuLN / self.EL, -self.nuTN / self.ET, 1 / self.EN]])
        
        S_shear = jnp.diag(jnp.array(
            [1. / (2. * self.muTN), 
             1. / (2. * self.muLN), 
             1./(2. * self.muLT)]
            )
            )  # [yz, xz, xy] order
        S_Mandel = jsc.linalg.block_diag(S_diag, S_shear)
        C_Mandel = jnp.linalg.inv(S_Mandel)
        return SymmetricTensor4(array = C_Mandel) 


class ElasticBehavior(SmallStrainBehavior):
    """A small strain linear elastic behavior."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""

    def constitutive_update(self, eps, state, dt):
        sig = self.elasticity.C @ eps
        new_state = state.update(strain=eps, stress=sig)
        return sig, new_state
