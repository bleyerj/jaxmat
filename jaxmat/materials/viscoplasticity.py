import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like
from jaxmat.state import (
    AbstractState,
    SmallStrainState,
    make_batched,
)
from jaxmat.tensors import SymmetricTensor2, dev
from .behavior import SmallStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import vonMises, AbstractPlasticSurface
from .viscoplastic_flows import (
    VoceHardening,
    NortonFlow,
    AbstractKinematicHardening,
    ArmstrongFrederickHardening,
)


class AFInternalState(SmallStrainState):
    """Internal state for the Armstrong-Frederick model"""

    p: float = eqx.field(converter=jnp.asarray, default=0.0)
    epsp: SymmetricTensor2 = eqx.field(default=SymmetricTensor2())
    a: SymmetricTensor2 = eqx.field(default=make_batched(SymmetricTensor2(), 2))


class AmrstrongFrederickViscoplasticity(SmallStrainBehavior):
    """
    Small-strain viscoplastic constitutive model with Armstrong-Frederick
    kinematic hardening, Voce isotropic hardening, and Norton-type viscous flow.

    This model represents a rate-dependent J2 (von Mises) viscoplastic material
    combining isotropic and kinematic hardening mechanisms under small strain
    assumptions. The total strain is additively decomposed into elastic and
    viscoplastic parts, and the evolution of the backstress follows the
    nonlinear Armstrong-Frederick law.

    Fields
    ----------
    elastic_model : LinearElasticIsotropic
        Linear isotropic elasticity defined by Young modulus and Poisson ratio.

    yield_stress : VoceHardening
        Isotropic hardening law controlling the evolution of the yield surface size.

    viscous_flow : NortonFlow
        Viscoplastic flow rule following Norton (power-law) viscosity formulation.

    kinematic_hardening : ArmstrongFrederickHardening
        Kinematic hardening model defining the backstress evolution rate with
        dynamic recovery (Armstrong-Frederick formulation).

    plastic_surface : vonMises
        J2-type yield (or loading) surface based on the deviatoric stress invariant.

    internal : AFInternalState
        Internal variables associated with the accumulated plastic strain and
        backstress tensor.

    Notes
    -----
    - The model is suitable for cyclic loading and ratcheting simulations.
    - When the viscous exponent tends to infinity, the model approaches the
      rate-independent limit.
    - The formulation assumes small strains and isotropic material symmetry.

    References
    ----------
    Armstrong, P. J., & Frederick, C. O. (1966).
        "A Mathematical Representation of the Multiaxial Bauschinger Effect for
        Hardening Materials." CEGB Report RD/B/N731.
    Voce, E. (1955).
        "A Practical Strain-Hardening Function." Metallurgia, 51, 219-226.
    Norton, F. H. (1929).
        "The Creep of Steel at High Temperatures." McGraw-Hill.
    """

    elastic_model: LinearElasticIsotropic
    yield_stress: VoceHardening
    viscous_flow: NortonFlow
    kinematic_hardening: ArmstrongFrederickHardening
    plastic_surface = vonMises()
    internal = AFInternalState()

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress
        sig_eq = lambda sig: self.plastic_surface(sig)

        def eval_stress(deps, dy):
            return sig_old + self.elastic_model.C @ (deps - dev(dy.epsp))

        def solve_state(deps, y_old):
            def residual(dy, args):
                y = tree_add(y_old, dy)
                sig = eval_stress(deps, dy)
                sig_eff = self.kinematic_hardening.sig_eff(sig, y.a)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(y.p)
                n = self.plastic_surface.normal(sig_eff)
                res = (
                    dy.p - dt * self.viscous_flow(yield_criterion),
                    dy.epsp - n * dy.p,
                    dy.a - self.kinematic_hardening(y.a, dy.p, dy.epsp),
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(
                residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint
            )
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GenericInternalState(SmallStrainState):
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    epsp: SymmetricTensor2 = SymmetricTensor2()
    nX: int = eqx.field(static=True, default=1)
    X: SymmetricTensor2 = eqx.field(init=False)

    def __post_init__(self):
        self.X = make_batched(SymmetricTensor2(), self.nX)


class GenericViscoplasticity(SmallStrainBehavior):
    elastic_model: LinearElasticIsotropic
    plastic_surface: AbstractPlasticSurface
    yield_stress: eqx.Module
    viscous_flow: eqx.Module
    kinematic_hardening: AbstractKinematicHardening
    internal: AbstractState = eqx.field(init=False)

    def __post_init__(self):
        self.internal = GenericInternalState(nX=self.kinematic_hardening.nvars)

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress
        sig_eq = lambda sig: self.plastic_surface(sig)

        def eval_stress(deps, dy):
            return sig_old + self.elastic_model.C @ (deps - dev(dy.epsp))

        def solve_state(deps, y_old):
            def residual(dy, args):
                y = tree_add(y_old, dy)
                sig = eval_stress(deps, dy)
                sig_eff = self.kinematic_hardening.sig_eff(sig, y.X)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(y.p)
                n = self.plastic_surface.normal(sig_eff)
                res = (
                    dy.p - dt * self.viscous_flow(yield_criterion),
                    dy.epsp - n * dy.p,
                    dy.X - self.kinematic_hardening(y.X, dy.p, dy.epsp),
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(
                residual, self.solver, dy0, has_aux=True, adjoint=self.adjoint
            )
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state
