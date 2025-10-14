# Composable and differentiable material models

Every material model in `jaxmat` inherits from [`equinox.Module`](https://docs.kidger.site/equinox/).  
An `eqx.Module` is a convenient extension of standard JAX **PyTrees**—nested data structures composed of tuples, lists, dictionaries, arrays, and other PyTrees—with the added benefit of behaving like lightweight Python classes.  

Equinox modules are:

- **JAX-compatible** (registered as PyTree nodes),
- **immutable** (frozen dataclasses), and
- **composable and differentiable** (supporting nesting of submodules).  

In effect, each material model is a structured container of differentiable parameters.

```{note}
State variables are also represented as `equinox.Module`. Other JAX-based packages like `diffrax` or `optimistix` also use them for representing solvers for instance.
---

## Hierarchical model composition

While internal state variables ($\boldsymbol{\alpha}$) and material parameters ($\boldsymbol{\theta}$) could in principle be flattened into a single large vector, in practice they are organized hierarchically into modules and submodules.  

For example:

- An *elastoplastic* model may be represented by a parent module containing:
  - an **elastic** submodule, and  
  - a **plastic** submodule.  
- The plastic submodule may itself include submodules for the **yield surface**, **hardening law**, and **flow rule**.

This modular structure promotes both clarity and reusability—complex constitutive models can be built from simple, well-defined components.

---

## Benefits of using Equinox PyTrees

- `equinox.Module` instances remain valid **PyTrees**, so they can be batched, mapped, or differentiated seamlessly using JAX transformations.  
- Functions such as `jax.vmap` or `jax.grad` can operate over the entire module hierarchy without special handling.  
- When fine-grained edits are needed (e.g. replacing a single subcomponent), standard PyTree utilities (like `jax.tree_map` or `optax.tree_utils`) can be used for selective modification.

For advanced manipulation, see the [Equinox documentation](https://docs.kidger.site/equinox/all-of-equinox/).

---

## Common Equinox patterns in `jaxmat`

1. **Automatic conversion to JAX arrays**  
   Many model attributes represent scalar or tensor-valued material parameters. These must be stored as `jax.Array` objects; plain Python floats or `numpy.ndarray` will not participate in JAX transformations or device placement.  
   To enforce this automatically, we use:

   ```python
   x: float = eqx.field(converter=jnp.asarray)
   ```

   This ensures the input (float, list, or NumPy array) is converted to a JAX-compatible array.

2. **Declarative defaults**  

    Default attribute values can be defined directly:

   ```python
   F: Tensor2 = eqx.field(default=Tensor2.identity())
   ```

   This avoids writing explicit `__init__` methods while keeping the model declarative.

3. **Safe JIT compilation with `eqx.filter_jit`**  
   We frequently wrap key methods (e.g. constitutive updates) with:

   ```python
   @eqx.filter_jit
   ```

   This decorator automatically filters out non-JAX types (such as solvers, configuration objects, or static metadata) from the JIT trace.  
   As a result, entire `equinox.Module` instances—potentially containing both static and dynamic fields—can be passed directly into JAX-compiled functions without causing tracing errors.
