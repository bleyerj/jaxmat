import jax
import equinox as eqx
import jax.numpy as jnp
from jaxmat.tensors import Tensor2, SymmetricTensor2


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


def tree_add(tree1, tree2):
    return jax.tree.map(lambda x, y: x + y, tree1, tree2)


def tree_zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


class SmallStrainState(AbstractState):
    internal: eqx.Module
    strain: SymmetricTensor2 = SymmetricTensor2()
    stress: SymmetricTensor2 = SymmetricTensor2()


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
