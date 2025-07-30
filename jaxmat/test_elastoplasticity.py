import jax
from time import time
import equinox as eqx
import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.loader import ImposedLoading, global_solve
from state import AbstractState, make_batched
from jaxmat.tensors import SymmetricTensor2
from viscoplastic_materials import Hosford
from elasticity import LinearElasticIsotropic
from elastoplasticity import GeneralIsotropicHardening, vonMisesIsotropicHardening


class SmallStrainState(AbstractState):
    strain: SymmetricTensor2 = SymmetricTensor2()
    stress: SymmetricTensor2 = SymmetricTensor2()
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    epsp: SymmetricTensor2 = SymmetricTensor2()


def test_elastoplasticity(material, Nbatch=1):

    state = make_batched(SmallStrainState(), Nbatch)
    Eps = state.strain

    plt.figure()
    Nsteps = 20
    eps_dot = 5e-3

    imposed_eps = 0
    dt = 0
    Nsteps = 30
    times = jnp.linspace(0, 4.0, Nsteps)
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
        Eps, state, stats = global_solve(Eps, state, loading, material, dt)
        num_steps = stats["num_steps"][0]
        print(
            f"Incr {i+1}: Num iter = {num_steps} Resolution time/iteration/batch:",
            (time() - tic) / num_steps / Nbatch,
        )

        Sig = state.stress

        plt.plot(Eps[0][0], Sig[0][0], "xb")
    plt.show()


E, nu = 200e3, 0.3
elastic_model = LinearElasticIsotropic(E, nu)

sig0 = 350.0
sigu = 500.0
b = 1e3


class YieldStress(eqx.Module):
    def __call__(self, p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))


Nbatch = int(1e3)

material = vonMisesIsotropicHardening(elastic_model, YieldStress())
test_elastoplasticity(material, Nbatch=Nbatch)

material = GeneralIsotropicHardening(
    elastic_model,
    Hosford(a=10.0),
    YieldStress(),
)
test_elastoplasticity(material, Nbatch=Nbatch)
