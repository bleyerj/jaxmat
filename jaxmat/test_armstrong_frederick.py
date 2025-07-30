import jax

# jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan
# jax.config.update("jax_traceback_filtering", "off")
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.tensors import dev, SymmetricTensor2
from time import time
import optimistix as optx
import equinox as eqx
from jaxmat.loader import ImposedLoading, global_solve
from jaxmat.materials.elasticity import LinearElasticIsotropic
from jaxmat.materials.viscoplasticity import (
    vonMises,
    ArmstrongFrederickHardening,
    VoceHardening,
    NortonFlow,
)
from state import AbstractState, make_batched


class State(AbstractState):
    strain: SymmetricTensor2 = SymmetricTensor2()
    stress: SymmetricTensor2 = SymmetricTensor2()
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    epsp: SymmetricTensor2 = SymmetricTensor2()
    a1: SymmetricTensor2 = SymmetricTensor2()
    a2: SymmetricTensor2 = SymmetricTensor2()


class AmrstrongFrederickViscoplasticity(eqx.Module):
    elastic_model: LinearElasticIsotropic
    plastic_surface: vonMises
    yield_stress: VoceHardening
    viscous_flow: NortonFlow
    kinematic_hardening: ArmstrongFrederickHardening
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def constitutive_update(self, eps, state, dt):
        eps_old = state.strain

        deps = eps - eps_old
        p_old = state.p
        epsp_old = state.epsp
        a_old1 = state.a1
        a_old2 = state.a2
        sig_old = state.stress
        sig_eq = lambda sig: jnp.clip(
            self.plastic_surface(sig), min=1e-8 * self.elastic_model.E
        )

        def compute_stress(deps, p_old, epsp_old, a_old1, a_old2):
            def residual(y, args):
                dp, depsp, da1, da2 = y
                da = jnp.asarray([da1, da2])
                a_old = jnp.asarray([a_old1, a_old2])
                a = a_old + da
                sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
                sig_eff = self.kinematic_hardening.sig_eff(sig, a)
                yield_criterion = sig_eq(sig_eff) - self.yield_stress(p_old + dp)
                n = self.plastic_surface.normal(sig_eff)
                res = (
                    dp - dt * self.viscous_flow(yield_criterion),
                    depsp - n * dp,
                )
                res = res + tuple(
                    da[i] - dp * (n - self.kinematic_hardening.g * a[i])
                    for i in range(2)
                )
                return res, sig

            y0 = (jnp.array(0.0), 0 * eps_old, 0 * a_old1, 0 * a_old2)
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp, depsp, da1, da2 = sol.value
            da = jnp.asarray([da1, da2])
            a_old = jnp.asarray([a_old1, a_old2])

            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            sig_eff = self.kinematic_hardening.sig_eff(sig, a_old + da)
            aux = (depsp, dp, da1, da2, sig_eff)
            return sig, aux

        sig, aux = compute_stress(deps, p_old, epsp_old, a_old1, a_old2)
        # Ct, aux = jax.jacfwd(compute_stress, has_aux=True)(deps, p_old, epsp_old, a_old)
        (depsp, dp, da1, da2, sig_eff) = aux
        new_state = state.add(strain=deps, p=dp, epsp=depsp, a1=da1, a2=da2)
        new_state = new_state.update(stress=sig)

        return sig, new_state


def test_explicit(Nbatch=1):
    E, nu = 200e3, 0.3
    elastic_model = LinearElasticIsotropic(E, nu)

    plastic_surface = vonMises()

    sig0 = 350.0
    sigu = 500.0
    b = 10.0
    yield_stress = VoceHardening(sig0, sigu, b)

    k = 15.0
    m = 2.0
    viscous_flow = NortonFlow(k, m)

    C = 15000.0
    g = 500.0
    kin_hardening = ArmstrongFrederickHardening(C, g)

    material = AmrstrongFrederickViscoplasticity(
        elastic_model, plastic_surface, yield_stress, viscous_flow, kin_hardening
    )

    state = make_batched(State(), Nbatch)
    Eps = state.strain

    plt.figure()
    eps_dot = 10e-4

    imposed_eps = 0

    t = 0
    dt = 0
    Nsteps = 100
    times = np.linspace(0, 20, Nsteps)

    results = [[Eps[0][0, 0], state.stress[0][0, 0], 0]]

    for dt in np.diff(times):
        t += dt
        sign = 1
        if t % 20 < 10:
            sign = 1
        else:
            sign = -1

        print("Time", t)
        imposed_eps += sign * eps_dot * dt

        # FIXME: need to start with non zero Eps
        def setxx(Eps):
            return SymmetricTensor2(tensor=Eps.tensor.at[0, 0].set(imposed_eps))

        Eps = eqx.filter_vmap(
            setxx,
        )(Eps)

        loading = ImposedLoading(epsxx=imposed_eps * jnp.ones((Nbatch,)))

        primals = (Eps, state, loading, material, dt)
        tangents = jax.tree.map(jnp.zeros_like, primals)
        tangents = eqx.tree_at(
            lambda t: t[2].strain_mask,
            tangents,
            jnp.zeros_like(loading.strain_mask, dtype=jax.float0),
        )
        tangents = eqx.tree_at(
            lambda t: t[3].kinematic_hardening.C,
            tangents,
            material.kinematic_hardening.C,
        )
        # tic = time()
        primals_out, tangents_out = eqx.filter_jvp(
            global_solve,
            primals,
            tangents,
        )

        Eps = primals_out[0]
        state = primals_out[1]
        dstate = tangents_out[1]

        Sig = state.stress
        dSig = dstate.stress
        results.append([Eps[0][0, 0], Sig[0][0, 0], Sig[0][0, 0] + dSig[0][0, 0]])
    results = np.asarray(results)
    plt.plot(results[:, 0], results[:, 1], "-", color="royalblue")
    plt.plot(results[:, 0], results[:, 2], "-", color="crimson")
    plt.show()


test_explicit(Nbatch=int(10))
