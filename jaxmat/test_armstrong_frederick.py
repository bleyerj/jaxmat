import jax

jax.config.update("jax_platform_name", "cpu")

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan
# jax.config.update("jax_traceback_filtering", "off")
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp
import equinox as eqx
from jaxmat.tensors import SymmetricTensor2
from jaxmat.loader import ImposedLoading, global_solve
import jaxmat.materials as jm
from time import time


def test_ArmstrongFrederick(Nbatch=1):
    E, nu = 200e3, 0.3
    elastic_model = jm.LinearElasticIsotropic(E, nu)

    sig0 = 350.0
    sigu = 500.0
    b = 10.0
    yield_stress = jm.VoceHardening(sig0, sigu, b)

    k = 15.0
    m = 2.0
    viscous_flow = jm.NortonFlow(k, m)

    C = 15000.0
    g = 500.0
    kin_hardening = jm.ArmstrongFrederickHardening(C, g)

    material = jm.AmrstrongFrederickViscoplasticity(
        elastic_model, yield_stress, viscous_flow, kin_hardening
    )

    state = material.init_state(Nbatch)
    Eps = state.strain

    plt.figure()

    eps_dot = 10e-4
    imposed_eps = 0

    t = 0
    dt = 0
    Nsteps = 100
    times = jnp.linspace(0, 20, Nsteps)

    results = [[Eps[0][0, 0], state.stress[0][0, 0], 0]]

    for i, dt in enumerate(jnp.diff(times)):
        t += dt
        sign = 1
        if t % 20 < 10:
            sign = 1
        else:
            sign = -1

        print("Time", t)
        imposed_eps += sign * eps_dot * dt

        # FIXME: need to start with non zero Eps
        def set_eps(Eps):
            return SymmetricTensor2(
                tensor=jnp.diag(jnp.asarray([1.0, -0.5, -0.5]) * imposed_eps)
            )

        Eps = eqx.filter_vmap(
            set_eps,
        )(Eps)
        tic = time()
        Sig, state = material.batched_constitutive_update(Eps, state, dt)
        print(
            f"Incr {i+1}: Resolution time/batch:",
            (time() - tic) / Nbatch,
        )

        # epsi = imposed_eps * jnp.ones((Nbatch,))
        # loading = ImposedLoading(
        #     epsxx=epsi,
        #     epsyy=-epsi / 2,
        #     epszz=-epsi / 2,
        #     epsxy=0 * epsi,
        #     epsxz=0 * epsi,
        #     epsyz=0 * epsi,
        # )

        # primals = (Eps, state, loading, material, dt)
        # tic = time()
        # Eps, state, stats = global_solve(*primals)
        # num_evals = (
        #     stats["num_steps"][0] + 1
        # )  # add 1 because we recompute residual at the end
        # print(
        #     f"Incr {i+1}: Num iter = {num_evals-1} Resolution time/iteration/batch:",
        #     (time() - tic) / num_evals / Nbatch,
        # )

        # tangents = jax.tree.map(jnp.zeros_like, primals)
        # tangents = eqx.tree_at(
        #     lambda t: t[2].strain_mask,
        #     tangents,
        #     jnp.zeros_like(loading.strain_mask, dtype=jax.float0),
        # )
        # tangents = eqx.tree_at(
        #     lambda t: t[3].kinematic_hardening.C,
        #     tangents,
        #     material.kinematic_hardening.C,
        # )
        # # tic = time()
        # primals_out, tangents_out = eqx.filter_jvp(
        #     global_solve,
        #     primals,
        #     tangents,
        # )

        # Eps = primals_out[0]
        # state = primals_out[1]
        # dstate = tangents_out[1]

        # Sig = state.stress
        # dSig = dstate.stress
        # results.append([Eps[0][0, 0], Sig[0][0, 0], Sig[0][0, 0] + dSig[0][0, 0]])
        Sig = state.stress
        results.append([Eps[0][0, 0], Sig[0][0, 0], 0])
    results = np.asarray(results)
    plt.plot(results[:, 0], results[:, 1], "-", color="royalblue")
    plt.plot(results[:, 0], results[:, 2], "-", color="crimson")
    plt.show()


test_ArmstrongFrederick(Nbatch=int(1))
