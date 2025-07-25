import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxmat.tensors import dev_vect as dev
from jaxmat.viscoplastic_materials import (
    LinearElasticIsotropic,
    AbstractPlasticSurface,
    vonMises,
)


def FB(x, y):
    return x + y - jnp.sqrt(x**2 + y**2)


class vonMisesIsotropicHardening(eqx.Module):
    elastic_model: LinearElasticIsotropic
    plastic_surface = vonMises()
    yield_stress: eqx.Module
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain

        deps = eps - eps_old
        p_old = state.p[0]  # convert to scalar
        epsp_old = state.epsp
        sig_old = state.stress
        mu = self.elastic_model.mu
        sig_el = sig_old + self.elastic_model.C @ deps
        sig_eq_el = self.plastic_surface(sig_el)
        n_el = self.plastic_surface.normal(sig_el)

        def compute_stress(deps, p_old, epsp_old):
            def residual(y, args):
                dp = y
                depsp = n_el * dp
                sig = sig_el - self.elastic_model.C @ dev(depsp)
                yield_criterion = (
                    sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)
                )
                res = FB(-yield_criterion / self.elastic_model.E, dp)
                return res, sig

            y0 = jnp.array(0.0)
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp = sol.value

            depsp = n_el * dp
            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            aux = (dp, depsp)
            return sig, aux

        sig, aux = compute_stress(deps, p_old, epsp_old)
        (dp, depsp) = aux
        state = state.add(strain=deps, p=dp, epsp=depsp)
        state = state.update(stress=sig)
        return sig, state


class GeneralIsotropicHardening(eqx.Module):
    elastic_model: LinearElasticIsotropic
    plastic_surface: AbstractPlasticSurface
    yield_stress: eqx.Module
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain

        deps = eps - eps_old
        p_old = state.p[0]  # convert to scalar
        epsp_old = state.epsp
        sig_old = state.stress

        def compute_stress(deps, p_old, epsp_old):
            def residual(y, args):
                dp, depsp = y
                sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
                yield_criterion = self.plastic_surface(sig) - self.yield_stress(
                    p_old + dp
                )
                n = self.plastic_surface.normal(sig)
                res = (
                    FB(-yield_criterion / self.elastic_model.E, dp),
                    depsp - n * dp,
                )
                return res, sig

            y0 = (jnp.array(0.0), jnp.zeros_like(eps_old))
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp, depsp = sol.value

            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            aux = (dp, depsp)
            return sig, aux

        sig, aux = compute_stress(deps, p_old, epsp_old)
        (dp, depsp) = aux
        state = state.add(strain=deps, p=dp, epsp=depsp)
        state = state.update(stress=sig)
        return sig, state
