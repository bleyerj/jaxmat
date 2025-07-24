import jax

# jax.config.update("jax_platform_name", "cpu")


import numpy as np
from dolfinx_materials.jax_materials import (
    LinearElasticIsotropic,
)
from dolfinx_materials.jax_materials.tensors import to_vect, dev
import matplotlib.pyplot as plt
import jax.numpy as jnp
from elastoplasticity import (
    vonMisesIsotropicHardening,
    GeneralIsotropicHardening,
    von_Mises_stress,
    Hosford_stress,
)
from time import time

print(jax.devices())
# raise


def residual(eps, sig, imposed_eps):
    return jnp.concatenate((jnp.atleast_1d(eps[0] - imposed_eps), sig[1:]))


def jacobian(eps, sig, Ct):
    Jac = jnp.zeros_like(Ct)
    Jac = Jac.at[0, 0].set(1)
    Jac = Jac.at[1:, :].set(Ct[1:, :])
    return Jac


def correct_eps(eps, sig, Ct, imposed_eps):
    res = residual(eps, sig, imposed_eps)
    Jac = jacobian(eps, sig, Ct)
    return eps + jnp.linalg.solve(Jac, -res)


global_residual = jax.vmap(jax.jit(residual), in_axes=(0, 0, None))
global_correct = jax.vmap(jax.jit(correct_eps), in_axes=(0, 0, 0, None))


def test_vonMises(Nbatch=1):
    E = 70e3
    sig0 = 350.0
    sigu = 500.0
    b = 1e3
    elastic_model = LinearElasticIsotropic(E=70e3, nu=0.3)

    def yield_stress(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))

    # material = vonMisesIsotropicHardening(elastic_model, yield_stress)
    material = GeneralIsotropicHardening(
        elastic_model,
        yield_stress,
        Hosford_stress,
        # lambda sig: jnp.sqrt(3 / 2.0) * jnp.linalg.norm(dev(sig), ord=10),
    )

    material.set_data_manager(Nbatch)

    plt.figure()
    Nsteps = 20
    eps_dot = 5e-3
    Eps = jnp.zeros((Nbatch, 6))
    sig = jnp.zeros_like(Eps)
    imposed_eps = 0

    dt = 0
    Nsteps = 30
    times = np.linspace(0, 5, Nsteps)
    t = 0
    for dt in np.diff(times):
        t += dt
        print("Time", t)
        imposed_eps += eps_dot * dt
        Eps = Eps.at[:, 0].set(imposed_eps)

        # Newton-Raphson solve
        nres = 1
        nres0 = jnp.max(jnp.linalg.norm(global_residual(Eps, sig, imposed_eps), axis=1))
        niter = 0
        atol = 1e-8
        rtol = 1e-8
        niter_max = 20
        while (nres > max(atol, rtol * nres0)) and (niter < niter_max):

            tic = time()
            sig, isv, Ct = material.integrate(Eps, dt)
            print("Integration time:", time() - tic)
            # tic = time()
            Eps = global_correct(Eps, sig, Ct, imposed_eps)
            # print("Correction time:", time() - tic)
            nres = jnp.max(
                jnp.linalg.norm(global_residual(Eps, sig, imposed_eps), axis=1)
            )

            niter += 1
            if niter >= niter_max:
                raise ValueError(f"Not converged within {niter_max} iterations.")

        material.data_manager.update()
        plt.plot(Eps[0][0], sig[0][0], "xb")
    plt.show()


test_vonMises(Nbatch=int(1e3))
