import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.state import (
    SmallStrainState,
    FiniteStrainState,
    make_batched,
    tree_add,
    tree_zeros_like,
)
from jaxmat.tensors import Tensor2, SymmetricTensor2
import pytest


class MyState(SmallStrainState):
    other_attribute: jax.Array = None


def test_state():
    state = MyState(internal=jnp.zeros((3,)))
    assert hasattr(state, "strain")
    assert hasattr(state, "stress")
    assert hasattr(state, "internal")
    assert hasattr(state, "other_attribute")

    ones = jnp.ones((3, 3))
    state = state.update(strain=ones)
    assert jnp.allclose(state.strain, ones)
    state = state.add(stress=2 * ones, strain=0.5 * ones)
    assert jnp.allclose(state.strain, 1.5 * ones)
    assert jnp.allclose(state.stress, 2 * ones)
    state = state.add(foobar=0)
    assert jnp.allclose(state.internal, jnp.zeros((3,)))
    assert state.other_attribute is None
    assert not hasattr(state, "foobar")


@pytest.mark.parametrize("Nbatch", [1, 10, 100])
def test_batching(Nbatch):
    state = MyState(internal=jnp.zeros((3,)))
    batched_state = make_batched(state, Nbatch)
    assert batched_state.strain.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.stress.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.internal.shape == (Nbatch, 3)
    assert batched_state.other_attribute is None

    state = MyState(
        internal=jnp.zeros((3, 3)),
    )
    batched_state = make_batched(state, Nbatch)
    assert batched_state.internal.shape == (Nbatch, 3, 3)


def test_tree_utils():
    class MyModule(eqx.Module):
        var: float = 0

    m1 = MyModule(var=1.0)
    m2 = MyModule(var=jnp.eye(3))
    m = tree_add(m1, m2)
    assert jnp.allclose(m.var, jnp.eye(3) + 1)
    m = tree_zeros_like(m2)
    assert jnp.allclose(m.var, jnp.zeros_like(m2.var))


def test_small_strain():
    state = SmallStrainState(strain=SymmetricTensor2.identity())
    state = state.update(stress=2 * state.strain)
    assert jnp.allclose(state.strain, jnp.eye(3))
    assert jnp.allclose(state.eps, jnp.eye(3))
    assert jnp.allclose(state.stress, 2 * jnp.eye(3))
    assert jnp.allclose(state.sig, 2 * jnp.eye(3))
    state = state.update(sig=3 * state.strain)
    assert jnp.allclose(state.stress, 3 * jnp.eye(3))
    assert jnp.allclose(state.sig, 3 * jnp.eye(3))


def test_finite_strain():
    state = FiniteStrainState()
    assert jnp.allclose(state.strain, jnp.eye(3))
    assert jnp.allclose(state.F, jnp.eye(3))
    state = state.update(stress=2 * state.strain)
    assert jnp.allclose(state.stress, 2 * jnp.eye(3))
    assert jnp.allclose(state.PK1, 2 * jnp.eye(3))
    state = state.update(PK1=3 * state.strain)
    assert jnp.allclose(state.stress, 3 * jnp.eye(3))
    assert jnp.allclose(state.PK1, 3 * jnp.eye(3))
