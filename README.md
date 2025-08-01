# `jaxmat`: Automated material constitutive modeling in JAX

## Statement of need

`jaxmat` is an open-source library for implementing material constitutive models in a way that integrates seamlessly with modern machine learning frameworks and existing finite element software. Built entirely in JAX, it leverages the broader JAX ecosystem -- particularly `equinox`, `diffrax`, and `optimistix` -- to provide a powerful and flexible modeling platform.

The library is designed around four core principles:

`jaxmat` offers an open-source library for implementing material constitutive models in way that is compatible with modern machine-learning frameworks and existing finite-element software. The library is implemented in JAX and heavily relies on additional features provided by the JAX ecosystem, in particular the `equinox`, `diffrax` and `optimistix` libraries. The design choices are based on the following premises:

1. **User-friendliness**: `jaxmat` provides domain-specific abstractions that simplify the definition of complex material models. It is not a database of hard-coded material behaviors with fixed parameters. Instead, it offers a generic and extensible framework for users to define, compose, and calibrate their own constitutive models with minimal boilerplate.

2. **Differentiability**: We fully exploit JAX’s *Automatic Differentiation* (AD) capabilities to eliminate the need for manually deriving consistent tangent operators, Jacobians of implicit systems, normal vectors to plastic surfaces, hyperelastic stress expressions, and more.
In `jaxmat`, every input and parameter of a constitutive model is differentiable, making it easy to compute sensitivities with respect to material or even algorithmic parameters! These features are essential for tasks such as material parameter identification, solving inverse problems, and performing uncertainty quantification.

3. **Modularity**: The library is designed to be highly modular, allowing users to easily mix and match modeling components. For example, in a generic elastoviscoplastic model, you can independently swap out the plastic yield surface, the hardening laws or the viscous flow, independently. Each component can also be replaced with a data-driven alternative, such as a neural network, without disrupting the overall structure.

4. **Efficiency**: `jaxmat` makes full use of JAX features such as *Just-In-Time compilation* (`jax.jit`), *automatic vectorization* (`jax.vmap`) and hardware acceleration (CPU or GPU through the XLA compiler). Constitutive models can be evaluated extremely efficiently and in parallel across batches of material points. Additionally, we have designed and selected algorithms specifically suited to the needs of computational mechanics for maximum performance, robustness, and generality, see the [Sharp bits]() section.

## Installation

Simply run a `pip` install and update `nvidia-cublas-cu12` as there are some issues with the version currently shipped with `jax-0.6.1`.

```bash
pip install .
pip install --upgrade nvidia-cublas-cu12
```
