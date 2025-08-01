import jax.numpy as jnp


def safe_sqrt(x, eps=1e-16):
    return jnp.sqrt(jnp.clip(x, min=eps))


def safe_norm(x, eps=1e-16):
    return jnp.sqrt(jnp.maximum(jnp.trace(x.T @ x), eps))


def FischerBurmeister(x, y):
    """Scalar Fischer-Burmeister function"""
    return x + y - safe_sqrt(x**2 + y**2)
