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
    """Abstract base class describing a mechanical behavior."""

    internal: eqx.AbstractVar[AbstractState]
    """Internal variables state."""
    solver: optx.AbstractRootFinder = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[0]
    )
    """Implicit solver."""
    adjoint: optx.AbstractAdjoint = eqx.field(
        static=True, init=False, default=DEFAULT_SOLVERS[1]
    )
    """Adjoint solver."""
    _batch_size: tuple = eqx.field(static=True, init=False, default=None)

    @abstractmethod
    def constitutive_update(self, inputs, state, dt):
        pass

    def batched_constitutive_update(self, inputs, state, dt):
        """Batched and jitted version of constitutive update along first axis of ``inputs`` and ``state``."""
        return eqx.filter_jit(
            eqx.filter_vmap(self.constitutive_update, in_axes=(0, 0, None))
        )(inputs, state, dt)

    def _init_state(self, cls, Nbatch=None):
        state = cls(internal=self.internal)
        if Nbatch is None:
            if self._batch_size is None:
                return cls(internal=self.internal)
            else:
                # Handle the case where the material has already been batched
                # we first batch cls without internals
                Nbatch = self._batch_size[0]
                state = make_batched(cls(), Nbatch)
                # we reaffect the already batched internals
                state = eqx.tree_at(lambda s: s.internal, state, self.internal)
                return state
        else:
            return make_batched(state, Nbatch)


class SmallStrainBehavior(AbstractBehavior):
    """Abstract small strain behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical small strain state."""
        return self._init_state(SmallStrainState, Nbatch)

    @abstractmethod
    def constitutive_update(self, eps, state, dt):
        """
        Perform the constitutive update for a given small strain increment
        for a small-strain behavior.

        This abstract method defines the interface for advancing the material
        state over a time increment based on the provided strain tensor.
        Implementations should return the updated stress tensor and internal
        variables, along with any auxiliary information required for consistent
        tangent computation or subsequent analysis.

        Parameters
        ----------
        eps : array_like
            Small strain tensor at the current integration point.
        state : PyTree
            PyTree containing the current state variables (stress, strain and internal) of the
            material.
        dt : float
            Time increment over which the update is performed.

        Returns
        -------
        stress : array_like
            Updated Cauchy stress tensor.
        new_state : PyTree
            Updated state variables after the constitutive update.

        Notes
        -----
        This method should be implemented by subclasses defining specific
        constitutive behaviors (elastic, plastic, viscoplastic, etc.).
        """
        pass


class FiniteStrainBehavior(AbstractBehavior):
    """Abstract finite strain behavior."""

    def init_state(self, Nbatch=None):
        """Initialize the mechanical finite strain state."""
        return self._init_state(FiniteStrainState, Nbatch)

    @abstractmethod
    def constitutive_update(self, F, state, dt):
        """
        Perform the constitutive update for a given deformation gradient increment
        for a finite-strain behavior.

        This abstract method defines the interface for advancing the material
        state over a time increment based on the provided strain tensor.
        Implementations should return the updated stress tensor and internal
        variables, along with any auxiliary information required for consistent
        tangent computation or subsequent analysis.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor at the current integration point.
        state : PyTree
            PyTree containing the current state variables (stress, strain and internal) of the
            material.
        dt : float
            Time increment over which the update is performed.

        Returns
        -------
        PK1 : array_like
            Updated first Piola-Kirchhoff stress tensor.
        new_state : PyTree
            Updated state variables after the constitutive update.

        Notes
        -----
        This method should be implemented by subclasses defining specific
        constitutive behaviors (elastic, plastic, viscoplastic, etc.).
        """
        pass
