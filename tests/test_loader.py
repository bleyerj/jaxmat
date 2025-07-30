import jax
import jax.numpy as jnp
from jaxmat.state import AbstractState, make_batched
from jaxmat.new_material_point import global_solve, make_imposed_loading, stack_loaders
from jaxmat.elasticity import LinearElasticIsotropic
from jaxmat.hyperelasticity import CompressibleNeoHookean, Hyperelasticity
from jaxmat.tensors import SymmetricTensor2, Tensor2, Tensor
import pytest


def test_small_strain():
    material = LinearElasticIsotropic(E=1e3, nu=0.3)

    class SmallStrainState(AbstractState):
        strain: jax.Array
        stress: jax.Array

        def __init__(self):
            self.strain = SymmetricTensor2()
            self.stress = SymmetricTensor2()

    loader1 = make_imposed_loading("small_strain", epsxx=0.02, sigxy=5.0)
    loader2 = make_imposed_loading("small_strain", sigxx=10.0)
    loader3 = make_imposed_loading("small_strain", epsyy=0.02 * jnp.ones((10,)))

    loaders = [loader1, loader2, loader3]
    loading = stack_loaders(loaders)
    dt = 0.1

    Nbatch = len(loading)
    state = make_batched(SmallStrainState(), Nbatch)
    eps0 = state.strain
    eps_sol, state_sol, stats = global_solve(eps0, state, loading, material, dt)
    stress = state_sol.stress
    assert jnp.allclose(
        stress[0], jnp.asarray([[20.0, 5.0, 0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert jnp.allclose(
        stress[1], jnp.asarray([[10.0, 0.0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    )
    assert jnp.allclose(
        stress[2], jnp.asarray([[0.0, 0.0, 0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
    )

    with pytest.raises(ValueError):
        make_imposed_loading("small_strain", epsXX=0.02)


def test_finite_strain():
    mu, lmbda = 1.0, 1e3
    material = Hyperelasticity(CompressibleNeoHookean(mu=mu, lmbda=lmbda))

    class FiniteStrain(AbstractState):
        strain: jax.Array
        stress: jax.Array
        PK2: jax.Array
        Cauchy: jax.Array

        def __init__(self):
            self.strain = Tensor2()
            self.stress = Tensor2()
            self.PK2 = SymmetricTensor2()
            self.Cauchy = SymmetricTensor2()

    lamb = 2.5
    lamb_ = 1 / jnp.sqrt(lamb)  # 1.0
    C = jnp.asarray([lamb**2, lamb_**2, lamb_**2])
    iC = 1 / C
    J = jnp.sqrt(jnp.prod(C))
    S = mu * (1 - iC) + lmbda * (J - 1) * J * iC
    sig = mu / J * (C - 1) + lmbda * (J - 1)

    loader1 = make_imposed_loading("finite_strain", FXX=lamb, FYY=lamb_, FZZ=lamb_)
    loader2 = make_imposed_loading(
        "finite_strain",
        FXX=lamb,
        FYY=lamb_,
        FZZ=lamb_,
        FXY=0,
        FYX=0,
        FXZ=0,
        FZX=0,
        FYZ=0,
        FZY=0,
    )
    lamb_list = jnp.full((10,), lamb)
    _lamb_list = 1 / jnp.sqrt(lamb_list)
    loader3 = make_imposed_loading(
        "finite_strain", FXX=lamb_list, FYY=_lamb_list, FZZ=_lamb_list
    )

    loading = stack_loaders([loader1, loader2, loader3])
    Nbatch = len(loading)

    state = make_batched(FiniteStrain(), Nbatch)
    F0 = state.strain
    dt = 0.0
    F_sol, state_sol, stats = global_solve(F0, state, loading, material, dt)

    assert jnp.allclose(state_sol.PK2[0], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[0], jnp.diag(sig))
    assert jnp.allclose(state_sol.PK2[1], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[1], jnp.diag(sig))
    assert jnp.allclose(state_sol.PK2[2], jnp.diag(S))
    assert jnp.allclose(state_sol.Cauchy[2], jnp.diag(sig))
