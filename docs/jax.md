# JAX implementation of material behaviors

```{image} images/jax_framework.png
:width: 600px
:align: center
```

The constitutive update of complex behaviors requires:

- looping over all quadrature points
- solving evolution equations for internal state variables
- computing the resulting stress
- computing the corresponding consistent tangent operator

A pure Python implementation generally prove extremely inefficient due to the loop over all quadrature points. To solve this issue, we will rely on the [JAX library](https://jax.readthedocs.io).

JAX is a Python library for accelerated (GPU) array computation and program transformation, designed for high-performance numerical computing and large-scale machine learning {cite:p}`jax2018github`. Its key features of interest here involve:

* [Accelerated `numpy`/`scipy` functions](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html)
* [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax-101/04-advanced-autodiff.html), see also the [AutoDiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
* [Just-In-Time compilation](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html) using `jax.jit`
* [Automatic Vectorization](https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html) using `jax.vmap`

All such features will prove extremely valuable when implementing optimized constitutive behavior models.

## A simple example

A `JAXMaterial` inherits from the general `Material` class and requires the user to implement the `constitutive_update` taking as arguments, the current strain `eps`, the previous state as a dictionary `state` and the time step `dt`.

As a simple example, here is an implementation of a linear elastic material:

```python
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial
from dolfinx_materials.material.jax.tensors import J, K


class LinearElasticIsotropic(JAXMaterial):
    def __init__(self, E, nu):
        super().__init__()
        E = 9 * kappa * mu / (3 * kappa + mu)
        nu = (3 * kappa - 2 * mu) / (2 * (3 * kappa + mu))
        self.C = 3 * self.kappa * J + 2 * self.mu * K

    def constitutive_update(self, eps, state, dt):
        sig = jnp.dot(self.C, eps)
        state["Stress"] = sig
        C_tang = self.C
        return C_tang, state
```

By default, the `state` dictionary contains two fields: `"Stress"` and `"Strain"` of dimension 6. For the sake of generality and simplicity, JAX behaviors are always written in a 3D setting, dimension `dim=6` corresponding to the 6 components of symmetric tensors.

We import the spherical and deviatoric projector tensors `J` and `K` from the `tensors` helper module and define the elastic stiffness operator `C`, see also [](tensors_conventions). The `constitutive_update` method simply computes $\boldsymbol{\sigma}=\mathbb{C}:\boldsymbol{\varepsilon}$. It then updates the state `"Stress"` with the computed values and outputs the tangent operator which is here simply `C` and the state containing the new stress.

Note that we explicitly use `jnp.dot` to perform the operation.

## JIT and automatic vectorization

To avoid a Python loop over all quadrature points, an alternative would be to rewrite the local constitutive function in vectorized form to work on a batch of strain-like quantities of shape `(dim, num_gauss)` where `num_gauss` is the total number of Gauss points. This strategy proves however error-prone and cumbersome in general, especially for complex behaviors incorporating logical branching such as plasticity.

Fortunately, JAX provides a way to automatically transform a function into an efficient vectorized form using `jax.vmap`. This means that any material behavior can be implemented as if working at a single material point.
In `JAXMaterial`, the method `batched_constitutive_update` is defined which handles all Gauss points simultaneously, it is defined as:

```python
self.batched_constitutive_update = jax.jit(jax.vmap(self.constitutive_update, in_axes=(0, 0, None))
```

The `in_axes` argument specifies that the vectorization occurs over axis 0 of the first two arguments (`eps` and `state`) but not the last one (`dt` is not batched). Note that JAX manages to do vectorization on all entries of the `state` dictionary.

Finally, the resulting function is also jitted using `jax.jit` for efficient compilation and execution by the XLA compiler.

## Automatic differentiation

In the above elastic example, the computation of the tangent operator is trivial. However, in most cases, it can be much more involved. In such cases, the library offers a way to do the computation seamlessly using *Automatic Differentiation* (AD) via the `@tangent_AD` decorator. To do so, simply decorate the `constitutive_update` method and returns the stress $\boldsymbol{\sigma}$ `sig` and the state `state` as outputs. Under the hood, this decorator does the following transformation:

```python
constitutive_update_tangent = jax.jacfwd(constitutive_update, argnums=0, has_aux=True)
```

The implementation relies on the `jax.jacfwd` function which computes the jacobian of the `constitutive_update` function with respect to argument `argnums=0`, namely $\boldsymbol{\varepsilon}$ here. The computation is done using forward-mode automatic differentiation, although here the tangent operator is of square shape so that there should be no significant difference with backward-mode. Finally, `constitutive_update` also returns `state` as an auxiliary output. This is specified explicitly with `has_aux=True` which indicates that the function returns a pair where the first element is considered the output of the mathematical function to be differentiated and the second element is auxiliary data. In this case, the output of `jacfwd` is `(C_tang, state)`.

The implementation of the `LinearElasticIsotropic` behavior would look in this case as:

```python
import jax.numpy as jnp
from dolfinx_materials.material.jax import JAXMaterial, tangent_AD
from dolfinx_materials.material.jax.tensors import J, K


class LinearElasticIsotropic(JAXMaterial):
    def __init__(self, E, nu):
        super().__init__()
        kappa = E / (3 * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.C = 3 * self.kappa * J + 2 * self.mu * K

    @tangent_AD
    def constitutive_update(self, eps, state, dt):
        sig = jnp.dot(self.C, eps)
        state["Stress"] = sig
        return sig, state
```

For more details on the use of AD on JAX behaviors, see [](jax_elastoplasticity.md) and the [](demos/jax/elastoplasticity/plane_elastoplasticity.md) demo.

## References

```{bibliography}
:filter: docname in docnames
```
