import jax
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD, JAXNewton
from dolfinx_materials.jax_materials.tensors import to_mat, dev
import optimistix as optx
from tensor_utils import eig33


def FB(x, y):
    return x + y - jnp.sqrt(x**2 + y**2)


@jax.jit
def von_Mises_stress(sig):
    return jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig))


@jax.jit
def Hosford_stress(sig, a=10):
    # Sig = to_mat(sig) + jnp.diag(jnp.asarray([1e-6, 2e-6, 3e-6]))
    sI = eig33(to_mat(sig))[0]
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


class AmrstrongFrederickViscoplasticity(JAXMaterial):

    def __init__(self, elastic_model, yield_stress, params):
        super().__init__()
        self.elastic_model = elastic_model
        self.yield_stress = yield_stress
        self.equivalent_stress = von_Mises_stress
        self.params = params
        self.solver = optx.Newton(rtol=1e-8, atol=1e-8)

    @property
    def internal_state_variables(self):
        return {"p": 1, "epsp": 6, "a": 12}

    def constitutive_update(self, eps, state, dt):
        eps_old = state["Strain"]

        deps = eps - eps_old
        p_old = state["p"][0]  # convert to scalar
        epsp_old = state["epsp"]
        a_old = state["a"]
        sig_old = state["Stress"]
        sig_eq = lambda sig: jnp.clip(
            self.equivalent_stress(sig), min=1e-8 * self.elastic_model.E
        )
        normal = lambda sig: 3 / 2 * dev(sig) / sig_eq(sig)

        def compute_stress(deps, p_old, epsp_old, a_old):

            C = self.elastic_model.C
            mu = self.elastic_model.mu

            def residual(y, args):
                dp, depsp, da = y
                sig = sig_old + C @ (deps - dev(depsp))
                a = a_old + da
                sig_eff = sig - 2 / 3 * self.params["C"] * jnp.sum(a_old + da, axis=0)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(p_old + dp)
                n = normal(sig_eff)
                res = (
                    dp
                    - dt
                    * jnp.maximum(yield_criterion / self.params["K"], 0)
                    ** self.params["m"],
                    depsp - n * dp,
                )
                res = res + tuple(
                    da[i] - dp * (n - self.params["g"] * a[i]) for i in range(2)
                )
                return res, sig

            y0 = (jnp.array(0.0), jnp.zeros_like(eps_old), jnp.zeros_like(a_old))
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp, depsp, da = sol.value

            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            sig_eff = sig - 2 / 3 * self.params["C"] * jnp.sum(a_old + da, axis=0)
            aux = (sig, depsp, dp, da, sig_eff)
            return sig, aux

        a_old = a_old.reshape(2, -1)
        Ct, aux = jax.jacfwd(compute_stress, has_aux=True)(deps, p_old, epsp_old, a_old)
        (sig, depsp, dp, da, sig_eff) = aux
        # jax.debug.print("sig={} dp={}, da={}, sig_eff={}", sig, dp, da, sig_eff)
        state["Strain"] += deps
        state["p"] += dp
        state["epsp"] += depsp
        state["a"] += da.flatten()
        state["Stress"] = sig
        return Ct, state
