import jax
import jax.numpy as jnp
from jaxmat.state import MechanicalState, make_batched
import pytest


class MyState(MechanicalState):
    additional_state: jax.Array

    def __init__(self, additional_state):
        super().__init__()
        self.additional_state = additional_state


def test_state():
    state = MyState(additional_state=jnp.zeros((3,)))
    assert hasattr(state, "strain")
    assert hasattr(state, "stress")
    assert hasattr(state, "additional_state")

    state = state.update(strain=jnp.ones((6,)))
    assert jnp.allclose(state.strain, jnp.ones((6,)))
    state = state.add(stress=2 * jnp.ones((6,)), strain=0.5 * jnp.ones((6,)))
    assert jnp.allclose(state.strain, 1.5 * jnp.ones((6,)))
    assert jnp.allclose(state.stress, 2 * jnp.ones((6,)))
    state = state.add(foobar=0)
    assert not hasattr(state, "foobar")


@pytest.mark.parametrize("Nbatch", [1, 10, 100])
def test_batching(Nbatch):
    state = MyState(additional_state=jnp.zeros((3,)))
    batched_state = make_batched(state, Nbatch)
    assert batched_state.strain.shape == (Nbatch, 6)
    assert batched_state.stress.shape == (Nbatch, 6)
    assert batched_state.additional_state.shape == (Nbatch, 3)

    state = MyState(
        additional_state=jnp.zeros((3, 3)),
    )
    batched_state = make_batched(state, Nbatch)
    assert batched_state.additional_state.shape == (Nbatch, 3, 3)
