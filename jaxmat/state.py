import jax
import equinox as eqx
import jax.numpy as jnp


class AbstractState(eqx.Module):
    def add(self, **changes):
        existing_keys = self.__dict__.keys()
        valid_changes = {k: v for k, v in changes.items() if k in existing_keys}

        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            [getattr(self, k) + v for k, v in valid_changes.items()],
        )

    def update(self, **changes):
        existing_keys = self.__dict__.keys()
        valid_changes = {k: v for k, v in changes.items() if k in existing_keys}

        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            list(valid_changes.values()),
        )


class MechanicalState(AbstractState):
    strain: jax.Array  # = eqx.field(default_factory=jnp.zeros((6,)))
    stress: jax.Array  # = eqx.field(default_factory=jnp.zeros((6,)))

    def __init__(self):
        self.stress = jnp.zeros((6,))
        self.strain = jnp.zeros((6,))


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
