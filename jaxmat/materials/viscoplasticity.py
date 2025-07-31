import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    SmallStrainState,
    tree_add,
    tree_zeros_like,
    make_batched,
)
from jaxmat.tensors import SymmetricTensor2, dev
from .behavior import SmallStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import vonMises
from .viscoplastic_flows import VoceHardening, NortonFlow, ArmstrongFrederickHardening


class InternalState(SmallStrainState):
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    epsp: SymmetricTensor2 = SymmetricTensor2()
    a: SymmetricTensor2 = make_batched(SymmetricTensor2(), 2)


class AmrstrongFrederickViscoplasticity(SmallStrainBehavior):
    elastic_model: LinearElasticIsotropic
    plastic_surface: vonMises
    yield_stress: VoceHardening
    viscous_flow: NortonFlow
    kinematic_hardening: ArmstrongFrederickHardening
    internal = InternalState()

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
                    dy.a
                    + dy.p * self.kinematic_hardening.g * y.a
                    - dy.p * make_batched(n, 2),
                )
                return res, y

            dy0 = tree_zeros_like(isv_old)
            sol = optx.root_find(residual, self.solver, dy0, has_aux=True)
            dy = sol.value
            y = sol.aux
            sig = eval_stress(deps, dy)
            return sig, y

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state
