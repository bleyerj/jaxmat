from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    AbstractState,
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)
from jaxmat.utils import default_value, enforce_dtype
from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4
import jaxmat.materials as jm
from jaxmat.materials.behavior import AbstractBehavior


import warnings
warnings.filterwarnings("ignore", message="Using `field\(init=False\)` on `equinox.Module`")

class Constant(eqx.Module):
    value: float | jax.Array = enforce_dtype()

    def __call__(self, **kwargs):
        return self.value

    def __repr__(self):
        return str(self.value)


def as_module(x):
    """Enforce a field to behave like a callable module. If not, wrap as a Constant."""
    if not callable(x):
        return Constant(x)
    else:
        return x


class LinearElasticIsotropic(jm.AbstractLinearElastic):
    """An isotropic linear elastic model."""

    E: eqx.Module = eqx.field(converter=as_module)
    r"""Young modulus $E$"""
    nu: eqx.Module = eqx.field(converter=as_module)
    r"""Poisson ratio $\nu$"""

    def C(self, **kwargs):
        return IsotropicTensor4(self.kappa(**kwargs), self.mu(**kwargs))

    def kappa(self, **kwargs):
        E = self.E(**kwargs)
        nu = self.nu(**kwargs)
        return E / (3 * (1 - 2 * nu))

    def mu(self, **kwargs):
        E = self.E(**kwargs)
        nu = self.nu(**kwargs)
        return E / (2 * (1 + nu))
    
    
class LinearElasticIsotropicThermal(LinearElasticIsotropic):
    """Isotropic linear elastic model with thermal expansion."""
    alpha: eqx.Module = eqx.field(converter=as_module)
    r"""Thermal expansion coefficient $\alpha$"""
    rho0: float
    r"""Initial density $\rho_0$"""
    cV: eqx.Module = eqx.field(converter=as_module)
    r"""Specific heat at constant strain $c_V$"""
    
    def thermal_stress_coeff(self, **kwargs):
        r"""Thermal stress coefficient $\kappa = \alpha \cdot E/(1-2\nu)$"""
        E = self.E(**kwargs)
        nu = self.nu(**kwargs)
        alpha = self.alpha(**kwargs)
        return alpha * E / (1 - 2 * nu)

class SmallStrainThermoMechanicalState(SmallStrainState):
    """State for small strain thermo-mechanical problems."""
    temperature: float = default_value(0.0)


class SmallStrainThermoMechanicalBehavior(AbstractBehavior):
    """Abstract small strain thermo-mechanical behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical small strain state."""
        return self._init_state(SmallStrainThermoMechanicalState, Nbatch)

    @abstractmethod
    def constitutive_update(self, inputs, state, dt):
        eps, T = inputs
        pass


class ElasticBehavior(SmallStrainThermoMechanicalBehavior):
    """Small strain linear elastic behavior with temperature-dependent properties."""

    elasticity: eqx.Module
    """The corresponding linear elastic model."""
    internal = None

    def constitutive_update(self, inputs, state, dt):
        """Update stress given strain and temperature."""
        eps, T = inputs
        sig = self.elasticity.C(T=T) @ eps
        new_state = state.update(strain=eps, stress=sig, temperature=T)
        return sig, new_state
    
    
class IsothermalElasticBehavior(SmallStrainThermoMechanicalBehavior):
    """Isothermal thermo-elastic behavior (no thermal diffusion).
    
    Temperature is prescribed externally and thermal expansion is accounted for.
    """
    
    elasticity: eqx.Module
    T0: float
    internal = None
    
    def init_state(self, Nbatch=None):
        """Initialize state at reference temperature T0."""
        state = self._init_state(SmallStrainThermoMechanicalState, Nbatch)
        return state.update(temperature=self.T0)
    
    def constitutive_update(self, inputs, state, dt):
        """Update stress accounting for thermal expansion."""
        eps, T = inputs
        
        # Thermo-elastic stress
        kappa_th = self.elasticity.thermal_stress_coeff(T=T)
        thermal_stress = kappa_th * (T - self.T0) * SymmetricTensor2.identity()
        
        sig = self.elasticity.C(T=T) @ eps - thermal_stress
        
        new_state = state.update(strain=eps, stress=sig, temperature=T)
        return sig, new_state
    
    
class AdiabaticThermoElasticBehavior(SmallStrainThermoMechanicalBehavior):
    """Adiabatic thermo-elastic behavior (no thermal diffusion).
    
    see The linear theory of thermoelasticity from the viewpoint of rational 
    thermomechanics. Ceramics- Silikaty, 49(4), 242-251.
    """
    
    elasticity: eqx.Module
    T0: float
    internal = None
    
    def init_state(self, Nbatch=None):
        """Initialize state at reference temperature T0."""
        state = self._init_state(SmallStrainThermoMechanicalState, Nbatch)
        return state.update(temperature=self.T0)
    
    def constitutive_update(self, inputs, state, dt):
        eps, _ = inputs# Temperature input ignored (internal variable)
        
        T_old = state.temperature
        eps_old = state.strain
        
        # Compute new temperature from entropy conservation, 
        kappa_th = self.elasticity.thermal_stress_coeff(T=T_old)
        # Current density (accounting for volume change)
        tr_eps = jnp.trace(eps.tensor)
        tr_eps_old = jnp.trace(eps_old.tensor)
        rho = self.elasticity.rho0 / (1 + tr_eps)
        dtr_eps = tr_eps - tr_eps_old
        cV_vol = rho * self.elasticity.cV(T=T_old)
        dT = -(kappa_th * self.T0) / cV_vol * dtr_eps
        
        T_new = T_old + dT
        
        # Thermo-elastic stress with updated temperature
        thermal_stress = kappa_th * (T_new - self.T0) * SymmetricTensor2.identity()
        sig = self.elasticity.C(T=T_new) @ eps - thermal_stress
        
        new_state = state.update(strain=eps, stress=sig, temperature=T_new)
        return sig, new_state


class YoungModulus(eqx.Module):
    E0 = 200e3
    a = 20.0

    def __call__(self, T):
        return self.E0 * (1 + self.a * T)


elasticity1 = LinearElasticIsotropic(E=200e3, nu=0.3)
elasticity2 = LinearElasticIsotropic(
    E=lambda T: 200e3 * (1 + 0.1 * T), nu=lambda T: 0.3
)
elasticity3 = LinearElasticIsotropic(E=YoungModulus(), nu=jnp.asarray(0.3))
elasticity4 = LinearElasticIsotropicThermal(
    E=200e3,
    nu=0.3,
    alpha=lambda T:2.31e-5 * (1 + 0.1 * T),
    rho0=7800.0,
    cV=lambda T:910e-6* (1 + 0.1 * T)
)

eps = SymmetricTensor2.identity() * 1e-3
for elasticity in [elasticity1, elasticity2, elasticity3, elasticity4]:
    material = ElasticBehavior(elasticity=elasticity)

    state = material.init_state()
    print(state.strain)
    print(state.temperature)

    
    inputs = eps, 1.5
    stress, new_state = material.constitutive_update(inputs, state, 0.0)
    print(stress.array)


material = IsothermalElasticBehavior(elasticity=elasticity4, T0=293.0)
state = material.init_state()
T = 303.0

stress, new_state = material.constitutive_update((eps, T), state, 0.0)
print(stress.array)

material = AdiabaticThermoElasticBehavior(
    elasticity=elasticity4,
    T0=293.0,
)

state = material.init_state()
print(f"T initial: {state.temperature} K")
stress, new_state = material.constitutive_update((eps, None), state, 1.0)

print(f"T final: {new_state.temperature} K")