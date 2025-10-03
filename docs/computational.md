# Computational aspects

`jaxmat` inherits all the standard features of the JAX ecosystem: automatic differentiation (AD), automatic vectorization (`vmap`), and just-in-time (JIT) compilation, while also adhering to JAX’s strict rules such as function purity, immutability of arrays, and program tracing. This section highlights aspects specific to `jaxmat`; for a general introduction to these concepts, see the [JAX documentation](https://docs.jax.dev/en/latest/).

## Precision

As material models generally involve relatively complex nonlinear equations to solve, we recommend working in `float64` precision for better stability and accuracy rather than in the default `float32` precision commonly used in machine learning applications. When importing `jaxmat`, `float64` precision is set by default. Precision can be set manually if needed as:

```{code-block} ipython
jax.config.update("jax_enable_x64", True)
```

## Data as PyTrees

Material models in `jaxmat` are represented as [`equinox.Module`](https://docs.kidger.site/equinox/) objects. These are a convenient extension of baseline JAX **PyTrees** (nested collections of tuples, lists, dicts, arrays, etc.), with the additional benefit of behaving like lightweight classes.

While internal state variables ($\balpha$) and material parameters ($\btheta$) could in principle be flattened into a single large vector, in practice they are organized hierarchically into modules and submodules.  
For example:

- an **elastoplastic model** may be represented by a parent module containing:
  - an *elastic* submodule, and
  - a *plastic* submodule.  
- the plastic submodule may itself contain submodules for the **yield surface**, **hardening law**, and **flow rule**.

This modular structure makes it easy to build complex models while keeping each component reusable. Moreover:

- `equinox` modules remain valid PyTrees, so they can be batched, mapped, and differentiated seamlessly.  
- `jax.vmap` can automatically apply an operation (e.g. constitutive update) across all leaves of the PyTree.  
- When fine-grained modifications are needed (e.g. replacing a single subcomponent), standard PyTree utilities can be used.  

For advanced manipulation of these data structures, see the [Equinox documentation](https://docs.kidger.site/equinox/all-of-equinox/).

## Just-In-Time Compilation and backend device

In JAX, it is usually recommended to JIT only the **outermost function** of a computation.  
In `jaxmat`, this means we typically JIT the **constitutive update function**, or a batched wrapper around it.  

When JIT-tracing, concrete values are replaced by *tracers*, and operations must remain compatible with JAX’s functional semantics. Care must be taken when writing implementations involving conditionals or loops to ensure that the traced code remains valid.

As in JAX, `jaxmat` supports device-portable batched constitutive updates; users may run the same code on CPU or GPU.
The observed performance will strongly depends on the hardware device, the used batch size  and the computational intensity of the material mode, see the [](demos/performance.md) demo for more details.

## On Automatic Vectorization

`jaxmat` relies heavily on **batched constitutive updates**. Instead of evaluating the stress for one strain input at a time, we can evaluate many inputs simultaneously — for example, one per quadrature point in a finite element assembly loop.  

This is achieved with `jax.vmap`, which transforms a function such as:

```{code-block} python
constitutive_update(material, strain, state, dt)
```

into a batched version:

```{code-block} python
batched_constitutive_update(material, strain_batch, state_batch, dt)
```

that operates efficiently across a whole array of strains or states. For more details, see the [](demos/batched_computation.md) demo.

It is also possible to batch through a set of material parameters sharing a common PyTree, see this other demo: [](demos/material_parameters_batching.md).

## On Automatic Differentiation

AD is central to `jaxmat`, most notably for computing the consistent tangent operator[^1]. In practice, AD is applied directly to the `constitutive_update` function.

However, the way the update is implemented matters. Consider implicit systems solved with a Newton method:

- If the Newton iterations are written out explicitly, AD will differentiate through all iterations (*algorithmic unrolling*). This leads to:

  - unnecessarily large and complex computational graphs,
  - possible numerical instability due to accumulation of floating-point errors.

- A better approach is to use the *implicit function theorem* (IFT): instead of differentiating through iterations, one solves an auxiliary linear system to obtain the derivative. This yields a more efficient and more accurate evaluation.

In `jaxmat`, we typically use solvers from `optimistix` and `diffrax` (the issue is the same for ODEs) which already implement this implicit differentiation technique.

[^1]: AD can also be used in other contexts, such as computing normal vectors to yield surfaces, derivatives of hyperelastic potentials, or Jacobians of implicit systems of nonlinear equations
