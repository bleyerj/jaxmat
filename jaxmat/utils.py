import jax
import equinox as eqx
import jax.numpy as jnp


def default_value(value, dtype=jnp.float64, **kwargs):
    """Initialize and convert a field with default `value` of imposed `dtype`."""
    return eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=dtype), default=value, **kwargs
    )
