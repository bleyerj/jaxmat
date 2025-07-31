import equinox as eqx
import optimistix as optx
from jaxmat.state import (
    SmallStrainState,
    FiniteStrainState,
    make_batched,
)


class SmallStrainBehavior(eqx.Module):
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    def get_state(self, Nbatch):
        return make_batched(SmallStrainState(internal=self.internal), Nbatch)


class FiniteStrainBehavior(eqx.Module):
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=optx.Newton(rtol=1e-8, atol=1e-8)
    )

    def get_state(self, Nbatch):
        return make_batched(FiniteStrainState(internal=self.internal), Nbatch)
