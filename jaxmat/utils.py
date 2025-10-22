import jax
import equinox as eqx
import jax.numpy as jnp


def default_value(value, dtype=jnp.float64, **kwargs):
    """Initialize and convert a field with default `value` of imposed `dtype`."""
    return eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=dtype), default=value, **kwargs
    )


def enforce_dtype(dtype=jnp.float64, **kwargs):
    """Initialize and convert a field with default `value` of imposed `dtype`."""
    return eqx.field(converter=lambda x: jnp.asarray(x, dtype=dtype), **kwargs)


def partition_by_node_names(model, freeze_names):
    """
    Partition an Equinox model into (trainable, static) where
    attributes listed in `freeze_names` are frozen (moved to static).
    """

    # Start with array-vs-nonarray partition
    trainable, static = eqx.partition(model, eqx.is_array)

    for name in freeze_names:
        sel = lambda m, name=name: getattr(m, name)

        # move out of trainable
        trainable = eqx.tree_at(
            sel, trainable, replace=None, is_leaf=lambda x: x is None
        )
        # copy original value into static
        static = eqx.tree_at(
            sel, static, replace=getattr(model, name), is_leaf=lambda x: x is None
        )

    return trainable, static
