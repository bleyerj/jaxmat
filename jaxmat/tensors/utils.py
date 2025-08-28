import jax.numpy as jnp
import optax


def safe_sqrt(x, eps=1e-16):
    nonzero_x = jnp.where(x > eps, x, eps)
    return jnp.where(x > eps, jnp.sqrt(nonzero_x), eps)


def safe_norm(x, eps=1e-16, **kwargs):
    return optax.safe_norm(x, eps, **kwargs)


def FischerBurmeister(x, y):
    """Scalar Fischer-Burmeister function"""
    return x + y - safe_sqrt(x**2 + y**2)
