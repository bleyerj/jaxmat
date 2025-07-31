import equinox as eqx
from jaxmat.state import (
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)


class SmallStrainBehaviour(eqx.Module):
    internal: None

    def get_state(self, Nbatch):
        return make_batched(SmallStrainState(internal=self.internal), Nbatch)


class FiniteStrainBehaviour(eqx.Module):
    internal: None

    def get_state(self, Nbatch):
        return make_batched(FiniteStrainState(internal=self.internal), Nbatch)
