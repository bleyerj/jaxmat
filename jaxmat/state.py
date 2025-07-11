import jax
import equinox as eqx
import jax.numpy as jnp


class AbstractState(eqx.Module):
    def add(self, **changes):
        return eqx.tree_at(
            lambda c: [c.__dict__[key] for key in changes.keys()],
            self,
            [self.__dict__[key] + val for key, val in changes.items()],
        )

    def update(self, **changes):
        keys, vals = zip(*changes.items())
        return eqx.tree_at(lambda c: [c.__dict__[key] for key in keys], self, vals)


class MechanicalState(AbstractState):
    strain: jax.Array  # = eqx.field(default_factory=jnp.zeros((6,)))
    stress: jax.Array  # = eqx.field(default_factory=jnp.zeros((6,)))


def make_batched(module: eqx.Module, Nbatch: int) -> eqx.Module:
    """Broadcasts all leaf arrays of a single unbatched module into a batched version.

    Args:
        module: An instance of an equinox Module (e.g., `State`) with array leaves.
        Nbatch: The number of batch items to broadcast.

    Returns:
        A new instance of the same class, with each array field having shape (Nbatch, ...).
    """

    def _broadcast(x):
        x_ = jnp.asarray(x)
        return jnp.broadcast_to(x_, (Nbatch,) + x_.shape)

    return jax.tree.map(_broadcast, module)
