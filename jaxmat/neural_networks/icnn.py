import equinox as eqx
import jax
import jax.numpy as jnp
from typing import List


def positive_param(W: jnp.ndarray) -> jnp.ndarray:
    # Example: enforce nonnegativity
    return jax.nn.softplus(W)


class ICNN(eqx.Module):
    """Input Convex Neural Network (ICNN).

    Implements a variant of the convex neural network architecture of Amos et al.

    Parameters
    ----------
    in_dim : int
        Input dimension (e.g. number of invariants).
    hidden_dims : list[int]
        Sequence of hidden layer widths.
    key : jax.random.PRNGKey
        Random key for parameter initialization.
    scale : float, optional
        Scaling factor for random initialization (default: 1.0).

    Fields
    ------
    Ws : list[jax.Array]
        Trainable weight matrices for each hidden layer. Constrained to be
        nonnegative via ``positive_param``.
    bs : list[jax.Array]
        Trainable bias vectors for each hidden layer.
    final_W : jax.Array
        Final layer weights combining last hidden layer output and input.

    Call
    ----
    __call__(x: jax.Array) -> jax.Array
        Forward pass. Given input of shape ``(..., in_dim)``, returns a scalar
        convex output representing the learned function.
    """

    Ws: List[jnp.ndarray]
    bs: List[jnp.ndarray]
    final_W: jnp.ndarray

    def __init__(
        self,
        in_dim: int,
        hidden_dims: List[int],
        key: jax.Array,
        scale: float = 1.0,
    ):
        keys = jax.random.split(key, len(hidden_dims) + 1)
        self.Ws, self.bs = [], []

        prev_dim = in_dim
        for i, h in enumerate(hidden_dims):
            k1 = keys[i]
            # Nonnegative W (hidden-to-hidden)
            W: jnp.ndarray = jax.random.normal(k1, (h, prev_dim)) * scale
            b: jnp.ndarray = jnp.zeros((h,))
            self.Ws.append(W)
            self.bs.append(b)
            prev_dim = h

        # Final layer: combines last hidden z and input x
        self.final_W = jax.random.normal(keys[-1], (1, prev_dim)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        z = x
        for W, b in zip(self.Ws, self.bs):
            W_pos = positive_param(W)
            z = jax.nn.softplus(z @ W_pos.T + b)
        return (z @ positive_param(self.final_W).T).squeeze()
