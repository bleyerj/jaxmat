import jax
import equinox as eqx
import jax.numpy as jnp
from jaxmat.tensors import Tensor2, SymmetricTensor2


class AbstractState(eqx.Module):
    def _resolve_aliases(self, changes):
        alias_map = getattr(self, "__alias_targets__", {})
        field_names = self.__dict__
        resolved = {}
        for k, v in changes.items():
            field_name = alias_map.get(k, k)
            if field_name in field_names:
                resolved[field_name] = v
        return resolved

    def add(self, **changes):
        valid_changes = self._resolve_aliases(changes)
        return eqx.tree_at(
            lambda c: [getattr(c, k) for k in valid_changes],
            self,
            [getattr(self, k) + v for k, v in valid_changes.items()],
        )

    def update(self, **changes):
        valid_changes = self._resolve_aliases(changes)
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
    internal: eqx.Module = None
    strain: SymmetricTensor2 = SymmetricTensor2()
    stress: SymmetricTensor2 = SymmetricTensor2()

    # define alias targets to authorize state updates with alias names
    __alias_targets__ = {"eps": "strain", "sig": "stress"}

    @property
    def eps(self):
        return self.strain

    @property
    def sig(self):
        return self.stress


class FiniteStrainState(AbstractState):
    internal: eqx.Module = None
    strain: Tensor2 = Tensor2().identity()
    stress: Tensor2 = Tensor2()

    # define alias targets to authorize state updates with alias names
    __alias_targets__ = {"F": "strain", "PK1": "stress"}

    @property
    def F(self):
        return self.strain

    @property
    def PK1(self):
        return self.stress


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
