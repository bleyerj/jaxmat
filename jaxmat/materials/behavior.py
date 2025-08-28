from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    AbstractState,
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)
from jaxmat.solvers import DEFAULT_SOLVERS


class AbstractBehavior(eqx.Module):
    internal: eqx.AbstractVar[AbstractState]
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[0]
    )
    adjoint: optx.AbstractAdjoint = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[1]
    )
    _batch_size: jax.Array = eqx.field(default=0.0, init=False, converter=jnp.asarray)

    @abstractmethod
    def constitutive_update(self, eps, state, dt):
        pass

    def batched_constitutive_update(self, eps, state, dt):
        return eqx.filter_jit(
            eqx.filter_vmap(self.constitutive_update, in_axes=(0, 0, None))
        )(eps, state, dt)

    def _init_state(self, cls, Nbatch=None):
        state = cls(internal=self.internal)
        if Nbatch is None:
            if (
                len(self._batch_size.shape) == 1
            ):  # Handle the case where the material has already been batched
                # we first batch cls without internals
                Nbatch = self._batch_size.shape[0]
                state = make_batched(cls(), Nbatch)
                # we reaffect the already batched internals
                state = eqx.tree_at(lambda s: s.internal, state, self.internal)
                return state
            else:
                return cls(internal=self.internal)
        else:
            return make_batched(state, Nbatch)


class SmallStrainBehavior(AbstractBehavior):
    def init_state(self, Nbatch=None):
        return self._init_state(SmallStrainState, Nbatch)


class FiniteStrainBehavior(AbstractBehavior):
    def init_state(self, Nbatch=None):
        return self._init_state(FiniteStrainState, Nbatch)
