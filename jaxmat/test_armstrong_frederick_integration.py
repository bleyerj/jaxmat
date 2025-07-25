import jax

# jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan
# jax.config.update("jax_traceback_filtering", "off")
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.tensors import SymmetricTensor4
from jaxmat.tensors import dev_vect as dev
from elastoviscoplasticity import AmrstrongFrederickViscoplasticity
from time import time
import optimistix as optx
from material_point_loading import (
    create_imposed_loading,
    create_loading_residual,
)

import equinox as eqx
from viscoplastic_materials import (
    vonMises,
    Hosford,
    VoceHardening,
    NortonFlow,
    ArmstrongFrederickHardening,
    LinearElasticIsotropic,
)
from state import MechanicalState, make_batched


class State(MechanicalState):
    p: jax.Array
    epsp: jax.Array
    a: jax.Array

    def __init__(self):
        super().__init__()
        self.p = jnp.zeros((1,))
        self.epsp = jnp.zeros((6,))
        self.a = jnp.zeros((12,))


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
        p_old = state.p[0]  # convert to scalar
        epsp_old = state.epsp
        a_old = state.a
        sig_old = state.stress
        sig_eq = lambda sig: jnp.clip(
            self.plastic_surface(sig), min=1e-8 * self.elastic_model.E
        )

        def compute_stress(deps, p_old, epsp_old, a_old):
            def residual(y, args):
                dp, depsp, da = y
                sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
                a = a_old + da
                sig_eff = self.kinematic_hardening.sig_eff(sig, a_old + da)
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

            y0 = (jnp.array(0.0), jnp.zeros_like(eps_old), jnp.zeros_like(a_old))
            sol = optx.root_find(residual, self.solver, y0, has_aux=True)
            dp, depsp, da = sol.value

            sig = sig_old + self.elastic_model.C @ (deps - dev(depsp))
            sig_eff = self.kinematic_hardening.sig_eff(sig, a_old + da)
            aux = (depsp, dp, da, sig_eff)
            return sig, aux

        a_old = a_old.reshape(2, -1)
        sig, aux = compute_stress(deps, p_old, epsp_old, a_old)
        # Ct, aux = jax.jacfwd(compute_stress, has_aux=True)(deps, p_old, epsp_old, a_old)
        (depsp, dp, da, sig_eff) = aux
        state = state.add(strain=deps, p=dp, epsp=depsp, a=da.flatten())
        state = state.update(stress=sig)

        return sig, state


def test_explicit(Nbatch=1):
    E, nu = 200e3, 0.3
    elastic_model = LinearElasticIsotropic(E, nu)

    plastic_surface = vonMises()  # osford()

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
    # material = make_batched(material, Nbatch)

    plt.figure()
    eps_dot = 10e-4

    Eps = jnp.zeros((Nbatch, 6))
    Sig = jnp.zeros_like(Eps)
    imposed_eps = 0

    t = 0
    dt = 0
    Nsteps = 100
    times = np.linspace(0, 20, Nsteps)

    def solve_mechanical_state(Eps, state, loading_data, material, dt):
        solver = optx.Newton(rtol=1e-8, atol=1e-8)
        residual = create_loading_residual(material)
        args = (loading_data, state, dt)
        sol = optx.root_find(
            residual, solver, Eps, args, has_aux=True
        )  # TODO: implement root find manually ?
        eps = sol.value
        sig, state = material.constitutive_update(
            eps, sol.aux, dt
        )  # evaluate one last time to compute state outside root_find which does not differentiate wrt auxiliary variables
        return eps, state

    global_solve = jax.jit(
        jax.vmap(solve_mechanical_state, in_axes=(0, 0, None, None, None))
    )

    results = []

    for dt in np.diff(times):
        t += dt
        sign = 1
        # if t % 20 < 10:
        #     sign = 1
        # else:
        #     sign = -1

        print("Time", t)
        imposed_eps += sign * eps_dot * dt

        Eps = Eps.at[:, 0].add(imposed_eps)
        material_ = jax.tree.map(jnp.zeros_like, material)

        loading_data = create_imposed_loading(epsxx=imposed_eps)

        primals = [Eps, state, loading_data, material, dt]
        tangents = jax.tree.map(jnp.zeros_like, primals)
        Eps_, state_, loading_data_, material_, dt_ = tangents
        material_ = eqx.tree_at(
            lambda t: t.yield_stress.sigu, material_, 0.5 * material.yield_stress.sigu
        )
        tangents[3] = material_

        tic = time()
        primals_out, tangents_out = eqx.filter_jvp(
            global_solve,
            primals,
            tangents,
        )
        print("Elapsed", time() - tic)
        Eps = primals_out[0]
        state = primals_out[1]
        dstate = tangents_out[1]

        Sig = state.stress
        print(Sig)
        dSig = dstate.stress

        results.append([Eps[0][0], Sig[0][0], Sig[0][0] + dSig[0][0]])
    results = np.asarray(results)
    plt.plot(results[:, 0], results[:, 1], "-", color="royalblue")
    plt.plot(results[:, 0], results[:, 2], "-", color="crimson")
    plt.show()


test_explicit(Nbatch=int(10))
