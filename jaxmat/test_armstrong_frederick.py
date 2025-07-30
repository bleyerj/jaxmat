import jax

# jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan
# jax.config.update("jax_traceback_filtering", "off")
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.tensors import dev, SymmetricTensor2
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
from jaxmat.state import SmallStrainState, make_batched, tree_add, tree_zeros_like


class InternalState(eqx.Module):
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    epsp: SymmetricTensor2 = SymmetricTensor2()
    a: SymmetricTensor2 = make_batched(SymmetricTensor2(), 2)


class AmrstrongFrederickViscoplasticity(eqx.Module):
    elastic_model: LinearElasticIsotropic
    plastic_surface: vonMises
    yield_stress: VoceHardening
    viscous_flow: NortonFlow
    kinematic_hardening: ArmstrongFrederickHardening
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    def get_state(self, Nbatch):
        return make_batched(SmallStrainState(internal=InternalState()), Nbatch)

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
    state = material.get_state(
        Nbatch
    )  # make_batched(SmallStrainState(internal=InternalState()), Nbatch)
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


test_explicit(Nbatch=int(1e2))
