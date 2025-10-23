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


def _rgetattr(obj, attr):
    """Like getattr, but supports dotted paths (e.g. 'layer.sub.weight')."""
    for name in attr.split("."):
        obj = getattr(obj, name)
    return obj


def partition_by_node_names(model, freeze_names):
    """
    Partition an Equinox model into (trainable, static) where
    attributes listed in `freeze_names` are frozen (moved to static).
    """

    # Start with array-vs-nonarray partition
    trainable, static = eqx.partition(model, eqx.is_array)

    for name in freeze_names:
        sel = lambda m, name=name: _rgetattr(m, name)

        # move out of trainable
        trainable = eqx.tree_at(
            sel, trainable, replace=None, is_leaf=lambda x: x is None
        )
        # copy original value into static
        static = eqx.tree_at(
            sel, static, replace=_rgetattr(model, name), is_leaf=lambda x: x is None
        )

    return trainable, static


def print_eqx_fields(obj, fields=None, indent=0):
    """
    Recursively print fields of an Equinox module or dataclass-like object.

    Args:
        obj: The Equinox module or object to inspect.
        fields: Optional list of field names (strings) to print.
                Supports nested paths like ["layer1", "layer2.weight"].
                If None, prints all fields recursively.
        indent: Internal indentation level (used for recursion).
    """
    pad = " " * indent

    # Helper to match top-level field names
    def matches(field_name):
        if fields is None:
            return True
        # Match if this field or any nested subpath starts with it
        return any(f == field_name or f.startswith(f"{field_name}.") for f in fields)

    if isinstance(obj, eqx.Module):
        print(f"{pad}{obj.__class__.__name__}:")
        for k, v in obj.__dict__.items():
            if not matches(k):
                continue  # skip fields not requested

            # Extract subfields relevant to this nested module (if any)
            subfields = None
            if fields is not None:
                subfields = [
                    f[len(k) + 1 :] for f in fields if f.startswith(f"{k}.")
                ] or None

            if isinstance(v, eqx.Module):
                print(f"{pad}  {k}:")
                print_eqx_fields(v, fields=subfields, indent=indent + 4)
            else:
                print(f"{pad}  {k} = {v}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            print(f"{pad}[{i}]: {v}")
    else:
        print(f"{pad}{obj}")
