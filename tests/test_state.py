import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.state import SmallStrainState, make_batched, tree_add, tree_zeros_like
import pytest


class MyState(SmallStrainState):
    internal: jax.Array


def test_state():
    state = MyState(internal=jnp.zeros((3,)))
    assert hasattr(state, "strain")
    assert hasattr(state, "stress")
    assert hasattr(state, "internal")

    ones = jnp.ones((3, 3))
    state = state.update(strain=ones)
    assert jnp.allclose(state.strain, ones)
    state = state.add(stress=2 * ones, strain=0.5 * ones)
    assert jnp.allclose(state.strain, 1.5 * ones)
    assert jnp.allclose(state.stress, 2 * ones)
    state = state.add(foobar=0)
    assert not hasattr(state, "foobar")


@pytest.mark.parametrize("Nbatch", [1, 10, 100])
def test_batching(Nbatch):
    state = MyState(internal=jnp.zeros((3,)))
    batched_state = make_batched(state, Nbatch)
    assert batched_state.strain.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.stress.tensor.shape == (Nbatch, 3, 3)
    assert batched_state.internal.shape == (Nbatch, 3)

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
