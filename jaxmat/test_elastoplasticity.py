import jax

jax.config.update("jax_platform_name", "cpu")

from time import time
import equinox as eqx
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.loader import ImposedLoading, global_solve
from jaxmat.tensors import SymmetricTensor2
import jaxmat.materials as jm


def test_elastoplasticity(material, Nbatch=1):

    state = material.init_state(Nbatch)
    Eps = state.strain

    plt.figure()

    eps_dot = 5e-3

    imposed_eps = 0
    dt = 0
    Nsteps = 11
    times = jnp.linspace(0, 1.0, Nsteps)
    t = 0
    for i, dt in enumerate(jnp.diff(times)):
        t += dt
        imposed_eps += eps_dot * dt

        # FIXME: need to start with non zero Eps
        def setxx(Eps):
            return SymmetricTensor2(tensor=Eps.tensor.at[0, 0].set(imposed_eps))

        Eps = eqx.filter_vmap(
            setxx,
        )(Eps)

        loading = ImposedLoading(epsxx=imposed_eps * jnp.ones((Nbatch,)))

        tic = time()
        Eps, state, stats = jax.block_until_ready(
            global_solve(Eps, state, loading, material, dt)
        )
        num_steps = stats["num_steps"][0]
        print(
            f"Incr {i+1}: Num iter = {num_steps} Resolution time/iteration/batch:",
            (time() - tic) / num_steps / Nbatch,
        )

        Sig = state.stress

        plt.plot(Eps[0][0], Sig[0][0], "xb")
    plt.show()


E, nu = 200e3, 0.3
elastic_model = jm.LinearElasticIsotropic(E, nu)

sig0 = 350.0
sigu = 500.0
b = 1e3


class YieldStress(eqx.Module):
    def __call__(self, p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))


Nbatch = int(1e4)

material = jm.vonMisesIsotropicHardening(elastic_model, YieldStress())
test_elastoplasticity(material, Nbatch=Nbatch)

# material = jm.GeneralIsotropicHardening(
#     elastic_model,
#     YieldStress(),
#     jm.Hosford(a=10.0),
# )
# test_elastoplasticity(material, Nbatch=Nbatch)
