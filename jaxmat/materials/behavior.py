from abc import abstractmethod
import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    AbstractState,
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)
from jaxmat.solvers import DEFAULT_SOLVER


class AbstractBehavior(eqx.Module):
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVER
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

    def get_state(self, Nbatch):
        return make_batched(SmallStrainState(internal=self.internal), Nbatch)


class FiniteStrainBehavior(AbstractBehavior):
    internal: eqx.AbstractVar[AbstractState]

    def get_state(self, Nbatch):
        return make_batched(FiniteStrainState(internal=self.internal), Nbatch)
