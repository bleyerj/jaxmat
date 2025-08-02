from abc import abstractmethod
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
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[0]
    )
    adjoint: optx.AbstractAdjoint = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[1]
    )

    @abstractmethod
    def constitutive_update(self, eps, state, dt):
        pass

    def batched_constitutive_update(self, eps, state, dt):
        return eqx.filter_jit(
            eqx.filter_vmap(self.constitutive_update, in_axes=(0, 0, None))
        )(eps, state, dt)


class SmallStrainBehavior(AbstractBehavior):
    internal: eqx.AbstractVar[AbstractState]

    def init_state(self, Nbatch=None):
        state = SmallStrainState(internal=self.internal)
        if Nbatch is None:
            return state
        else:
            return make_batched(state, Nbatch)


class FiniteStrainBehavior(AbstractBehavior):
    internal: eqx.AbstractVar[AbstractState]

    def init_state(self, Nbatch=None):
        state = FiniteStrainState(internal=self.internal)
        if Nbatch is None:
            return state
        else:
            return make_batched(state, Nbatch)
