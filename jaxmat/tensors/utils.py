import jax.numpy as jnp
import optax


def safe_fun(fun, x, norm=None, eps=1e-16):
    if norm is None:
        norm = lambda x: x
    nonzero_x = jnp.where(norm(x) > eps, x, 0 * x)
    return jnp.where(norm(x) > eps, fun(nonzero_x), 0)


def safe_sqrt(x, eps=1e-16):
    nonzero_x = jnp.where(x > eps, x, eps)
    return jnp.where(x > eps, jnp.sqrt(nonzero_x), eps)


def safe_norm(x, eps=1e-16, **kwargs):
    return optax.safe_norm(x, eps, **kwargs)


def FischerBurmeister(x, y):
    """Scalar Fischer-Burmeister function"""
    return x + y - safe_sqrt(x**2 + y**2)
