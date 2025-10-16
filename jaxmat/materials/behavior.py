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
    _batch_size: jax.Array = eqx.field(default=0, init=False, converter=jnp.asarray)

    @abstractmethod
    def constitutive_update(self, eps, state, dt):
        """
        Perform the constitutive update for a given strain increment.

        This abstract method defines the interface for advancing the material
        state over a time increment based on the provided strain tensor.
        Implementations should return the updated stress tensor and internal
        variables, along with any auxiliary information required for consistent
        tangent computation or subsequent analysis.

        Parameters
        ----------
        eps : array_like
            Strain tensor at the current integration point.
            Shape and convention depend on the model implementation (e.g., small
            strain vector form or finite strain tensor form).
        state : PyTree
            PyTree containing the current state variables (stress, strain and internal) of the
            material.
        dt : float
            Time increment over which the update is performed.

        Returns
        -------
        stress : array_like
            Updated Cauchy or Kirchhoff stress tensor corresponding to `eps`.
        new_state : PyTree
            Updated state variables after the constitutive update.

        Notes
        -----
        This method should be implemented by subclasses defining specific
        constitutive behaviors (elastic, plastic, viscoplastic, etc.).
        """
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
