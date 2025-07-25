import jax


from time import time
import numpy as np
import equinox as eqx
import optimistix as optx
import matplotlib.pyplot as plt
import jax.numpy as jnp

from material_point_loading import (
    create_imposed_loading,
    create_loading_residual,
)
from state import MechanicalState, make_batched

from viscoplastic_materials import LinearElasticIsotropic, vonMises, Hosford
from elastoplasticity import GeneralIsotropicHardening, vonMisesIsotropicHardening


class State(MechanicalState):
    p: jax.Array
    epsp: jax.Array

    def __init__(self):
        super().__init__()
        self.p = jnp.zeros((1,))
        self.epsp = jnp.zeros((6,))


def test_vonMises(Nbatch=1):
    E, nu = 200e3, 0.3
    elastic_model = LinearElasticIsotropic(E, nu)

    sig0 = 350.0
    sigu = 500.0
    b = 1e3

    class YieldStress(eqx.Module):
        def __call__(self, p):
            return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    material = GeneralIsotropicHardening(
        elastic_model,
        Hosford(),
        YieldStress(),
    )
    # material = vonMisesIsotropicHardening(elastic_model, YieldStress())

    state = make_batched(State(), Nbatch)

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
        return eps, state, sol.stats

    global_solve = jax.jit(
        jax.vmap(solve_mechanical_state, in_axes=(0, 0, None, None, None))
    )

    plt.figure()
    Nsteps = 20
    eps_dot = 5e-3
    Eps = jnp.zeros((Nbatch, 6))
    Sig = jnp.zeros_like(Eps)
    imposed_eps = 0

    plt.plot(Eps[0][0], Sig[0][0], "xb")

    dt = 0
    Nsteps = 30
    times = np.linspace(0, 4.0, Nsteps)
    t = 0
    for i, dt in enumerate(np.diff(times)):
        t += dt
        # print("Time", t)
        imposed_eps += eps_dot * dt
        Eps = Eps.at[:, 0].set(imposed_eps)
        loading_data = create_imposed_loading(epsxx=imposed_eps)

        tic = time()
        Eps, state, stats = global_solve(Eps, state, loading_data, material, dt)
        num_steps = stats["num_steps"][0]
        print(
            f"Incr {i+1}: Num iter = {num_steps} Resolution time/iteration:",
            (time() - tic) / num_steps,
        )

        Sig = state.stress

        plt.plot(Eps[0][0], Sig[0][0], "xb")
    plt.show()


test_vonMises(Nbatch=int(1e3))
