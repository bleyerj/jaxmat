import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton
from dolfinx_materials.jax_materials.tensors import to_mat, dev
import optimistix as optx
from tensor_utils import eig33, jacobi_eig_3x3


def FB(x, y):
    return x + y - jnp.sqrt(x**2 + y**2)


@jax.jit
def von_Mises_stress(sig):
    return jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig))


@jax.jit
def Hosford_stress(sig, a=10):
    # Sig = to_mat(sig) + jnp.diag(jnp.asarray([1e-6, 2e-6, 3e-6]))
    sI = jacobi_eig_3x3(to_mat(sig))[0]
    return (
        1
        / 2
        * (
            jnp.abs(sI[0] - sI[1]) ** a
            + jnp.abs(sI[0] - sI[2]) ** a
            + jnp.abs(sI[2] - sI[1]) ** a
        )
    ) ** (1 / a)


class vonMisesIsotropicHardening(JAXMaterial):
    def __init__(self, elastic_model, yield_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = jax.jit(yield_stress)
        self.equivalent_stress = von_Mises_stress
        self.solver = optx.Newton(rtol=1e-8, atol=1e-8)

    @property
    def internal_state_variables(self):
        return {"p": 1}

    # @tangent_AD
    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]
        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        sig_old = state["Stress"]

        def compute_stress(deps, p_old):

            C = self.elastic_model.C
            mu = self.elastic_model.mu
            sig_el = sig_old + C @ deps
            sig_eq_el = jnp.clip(
                self.equivalent_stress(sig_el), a_min=1e-8 * self.elastic_model.E
            )
            n_el = dev(sig_el) / sig_eq_el

            def residual(dp, args):
                depsp = 3 / 2 * n_el * dp
                sig = sig_el - C @ depsp
                yield_criterion = (
                    sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)
                )
                return FB(-yield_criterion, dp), sig

            sol = optx.root_find(residual, self.solver, 0.0, has_aux=True)
            dp = sol.value
            sig = sol.aux

            depsp = 3 / 2 * n_el * dp
            sig = sig_old + self.elastic_model.C @ (deps - depsp)
            aux = (sig, depsp, dp)
            return sig, aux

        Ct, aux = jax.jacfwd(compute_stress, has_aux=True)(deps, p_old)
        (sig, depsp, dp) = aux

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        return Ct, state


class GeneralIsotropicHardening(JAXMaterial):

    def __init__(self, elastic_model, yield_stress, equivalent_stress):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_stress = equivalent_stress
        self.solver = optx.Newton(rtol=1e-8, atol=1e-8)

    @property
    def internal_state_variables(self):
        return {"p": 1}

    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]

        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        sig_old = state["Stress"]

        normal = jax.jacfwd(self.equivalent_stress)

        def compute_stress(deps, p_old):

            C = self.elastic_model.C
            mu = self.elastic_model.mu
            sig_el = sig_old + C @ deps
            sig_eq_el = jnp.clip(
                self.equivalent_stress(sig_el), min=1e-8 * self.elastic_model.E
            )

            def residual(y, args):
                dp, depsp = y
                sig = sig_el - C @ dev(depsp)
                yield_criterion = (
                    sig_eq_el - 3 * mu * dp - self.yield_stress(p_old + dp)
                ) / self.elastic_model.E
                res = (FB(-yield_criterion, dp), depsp - normal(sig) * dp)
                return res, sig

            y0 = (jnp.array(0.0), jnp.zeros_like(eps_old))
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp, depsp = sol.value

            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            aux = (sig, depsp, dp)
            return sig, aux

        Ct, aux = jax.jacfwd(compute_stress, has_aux=True)(deps, p_old)
        (sig, depsp, dp) = aux

        state["Strain"] += deps
        state["p"] += dp
        state["Stress"] = sig
        return Ct, state
