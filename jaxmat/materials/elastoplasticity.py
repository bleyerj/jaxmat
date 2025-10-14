import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from optax.tree_utils import tree_add, tree_zeros_like
from jaxmat.utils import default_value
from jaxmat.state import AbstractState
from jaxmat.tensors import SymmetricTensor2, dev
from .behavior import SmallStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import (
    AbstractPlasticSurface,
    vonMises,
)
from jaxmat.tensors.utils import FischerBurmeister as FB
import jax


class InternalState(AbstractState):
    """Internal state for hardening plasticity"""

    p: jax.Array = default_value(0.0)
    epsp: SymmetricTensor2 = eqx.field(default_factory=lambda: SymmetricTensor2())


class vonMisesIsotropicHardening(SmallStrainBehavior):
    elastic_model: LinearElasticIsotropic
    yield_stress: eqx.Module
    plastic_surface: AbstractPlasticSurface = vonMises()
    internal: AbstractState = InternalState()

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def solve_state(deps, isv_old):
            mu = self.elastic_model.mu
            sig_el = sig_old + self.elastic_model.C @ deps
            sig_eq_el = self.plastic_surface(sig_el)
            n_el = self.plastic_surface.normal(sig_el)
            p_old = isv_old.p

            def residual(dp, args):
                p = p_old + dp
                yield_criterion = sig_eq_el - 3 * mu * dp - self.yield_stress(p)
                res = FB(-yield_criterion / self.elastic_model.E, dp)
                return res

            dy0 = jnp.array(0.0)
            sol = optx.root_find(residual, self.solver, dy0, adjoint=self.adjoint)
            dp = sol.value

            depsp = n_el * dp
            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            isv = isv_old.add(p=dp, epsp=depsp)
            return sig, isv

        sig, isv = solve_state(deps, isv_old)

        new_state = state.update(strain=eps, stress=sig, internal=isv)
        return sig, new_state


class GeneralIsotropicHardening(SmallStrainBehavior):
    elastic_model: LinearElasticIsotropic
    yield_stress: eqx.Module
    plastic_surface: AbstractPlasticSurface
    internal = InternalState()

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain
        deps = eps - eps_old
        isv_old = state.internal
        sig_old = state.stress

        def eval_stress(deps, dy):
            return sig_old + self.elastic_model.C @ (deps - dy.epsp)

        def solve_state(deps, y_old):
            p_old = y_old.p

            def residual(dy, args):
                dp, depsp = dy.p, dy.epsp
                sig = eval_stress(deps, dy)
                yield_criterion = self.plastic_surface(sig) - self.yield_stress(
                    p_old + dp
                )
                n = self.plastic_surface.normal(sig)
                res = (
                    FB(-yield_criterion / self.elastic_model.E, dp),
                    depsp - n * dp,
                )
                y = tree_add(y_old, dy)
                return (res, y)

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
