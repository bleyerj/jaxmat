# General framework

The `dolfinx_materials` library aims at providing a general framework for using custom material behaviors within the `FEniCSx` PDE library.
In particular, `ufl` and `dolfinx` limitations make it difficult to formulate complex material behaviors such as viscoelasticity, plasticity, damage or even some complex hyperelastic behaviors.

The library expands upon previous implementations of user-defined constitutive models by providing a more general and extensible setting, targeting advanced implementation of material models including:

- materials defined by internal state variables and a system of implicit equations
- multi-physics behaviors not limited to mechanics e.g. thermo-hydro-mechanics models
- material behaviors defined by machine-learning or data-driven methods
- multiscale material behaviors

```{seealso}
We list below a few resources upon which this library has been built:

- the [`mgis.fenics`](https://thelfer.github.io/mgis/web/mgis_fenics.html) project, binding `MFront` with legacy `FEniCS`
- [implementation of elasto-plasticity](https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html) within `FEniCSx`
- [JAX implementation of viscoelasticity](https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/linear_viscoelasticity_jax/linear_viscoelasticity_jax.html) within `FEniCSx`

```

```{admonition} Future changes
:class: important

Integration with the `dolfinx` library will probably be reimplemented in the next future based on the concept of `ExternalOperator`. For more details, see [External operators within FEniCSx/DOLFINx](https://a-latyshev.github.io/dolfinx-external-operator/).
```

## Nonlinear constitutive equations
$\newcommand{\bsig}{\boldsymbol{\sigma}}
\newcommand{\beps}{\boldsymbol{\varepsilon}}
\newcommand{\balpha}{\boldsymbol{\alpha}}
\newcommand{\bepsv}{\boldsymbol{\varepsilon}^\text{v}}
\newcommand{\epsv}{\varepsilon^\text{v}}
\newcommand{\sigv}{\sigma^\text{v}}
\newcommand{\bu}{\boldsymbol{u}}
\newcommand{\bq}{\boldsymbol{q}}
\newcommand{\bv}{\boldsymbol{v}}
\newcommand{\bI}{\boldsymbol{I}}
\newcommand{\bF}{\boldsymbol{F}}
\newcommand{\bP}{\boldsymbol{P}}
\newcommand{\CC}{\mathbb{C}}
\newcommand{\bT}{\boldsymbol{T}}
\newcommand{\dOm}{\,\text{d}\Omega}
\newcommand{\dS}{\,\text{d}S}
\newcommand{\dt}{\,\text{d}t}
\newcommand{\Neumann}{{\partial \Omega_\text{N}}}
\newcommand{\Dirichlet}{{\partial \Omega_\text{D}}}$

In the following, we aim at solving a small-strain solid mechanics problem involving a general nonlinear constitutive behavior. Such a constitutive behavior will be seen as generic black-box function. It expresses the stress $\bsig$ as a function of the total strain $\beps$, yielding the following nonlinear variational problem:

```{math}
:label: nonlinear-variational-model

\int_{\Omega} \bsig(\beps):\nabla^s \bv \dOm = \int_\Omega \boldsymbol{f}\cdot\bv \dOm + \int_{\partial \Omega_\text{N}} \bT\cdot\bv \dS \quad \forall\bv \in V
```

In practice, the nonlinear function $\bsig(\beps)$ is parameterized by material parameters and can also depend upon a set of variables which reflects the state and past history of the material. Such **state** consists of so-called *internal state variables* which may be plastic strains, viscous strains/stresses, damage, porosity, etc. We collect them in the global notation $\balpha$.

The mapping $\bsig(\beps)$ is therefore often implicit and would rather be something like:
```{math}
\bsig(\beps,\balpha) \quad \text{s.t.}\quad F(\beps,\balpha,\dot{\balpha})=0
```
where $F(\beps,\balpha,\dot{\balpha})$ defines a system of evolution equations of the internal state variables. Upon performing a time-discretization, we collect in the set $\mathcal{S}_n$ the state of the material at time $t_n$, here we have $\mathcal{S}_n = \{\balpha_n\}$. The evaluation of the constitutive equation will be strain-driven in the sense that we provide a total strain increment $\Delta \beps$ and the previous state at time $t_n$ and the role of the constitutive model is to compute the new stress $\bsig_{n+1}$ and the new state $\mathcal{S}_{n+1} = \{\balpha_{n+1}\}$:

```{math}
:label: constitutive-black-box

\beps=\beps_n+\Delta\beps, \mathcal{S}_n \longrightarrow \boxed{\text{CONSTITUTIVE RELATION}}\longrightarrow \bsig_{n+1}, \mathcal{S}_{n+1}
```
where $\bsig_{n+1},\balpha_{n+1}$ are obtained by solving the following problem:
```{math}
\bsig_{n+1}=\bsig(\beps_n+\Delta\beps,\balpha_{n+1}) \quad \text{s.t.}\quad F_n(\beps_n+\Delta\beps,\balpha_{n+1})=0
```
given $\Delta\beps, \beps_n, \balpha_n$ and where $F_n$ denotes a time-discretized version of $F$ i.e. depending on $\balpha_n$ for instance. For more details on the various discretization strategies and solving schemes, we refer to {cite:p}`simo2006computational`. In the `dolfinx_materials` library, such an evolution will be managed by a concrete implementation of a `Material` object using, for instance, a third-party library. This constitutive update is therefore fully decoupled from the finite-element library which sees only $\bsig(\beps)$ as an abstract non-linear function.

`FEniCSx`, through UFL, is well-equipped to handle relatively complex constitutive behaviors, such as hyperelasticity (including anisotropic models), by defining them via a free energy potential and mathematical primitives. In these cases, *symbolic* automatic differentiation (AD) is efficiently employed to derive stress-strain relationships directly. However, UFL exhibits limitations when it comes to constitutive models defined by implicit expressions, such as those involving ordinary differential equations (ODEs) for auxiliary state variables. The fundamental distinction here is that such constitutive laws must be conceptualized as computational algorithms rather than closed-form mathematical expressions, highlighting the difference between *symbolic* and *algorithmic* AD.

## Consistent tangent operator

However, when solving the *global* variational problem {eq}`nonlinear-variational-model`, we generally use a *global* Newton-Raphson solver. The jacobian of this nonlinear problem involves the following tangent bilinear form:

```{math}
:label: tangent-bilinear-form

a_\text{tangent}(\bu,\bv) = \int_{\Omega} \nabla^s \bu: \mathbb{C}_\text{tang}(\beps):\nabla^s \bv \dOm \quad \text{where } \mathbb{C}_\text{tang}(\beps)=\dfrac{\partial \bsig}{\partial \beps}(\beps)
```

which involves the local tangent stiffness tensor $\mathbb{C}_\text{tang}(\beps)$. The latter is the direct differentiation of the mapping {eq}`constitutive-black-box`. The `Material` object must therefore, not only provide a concrete implementation of the stress evaluation but also of its derivative with respect to the imposed strain $\beps$. This step is crucial as it directly impacts the convergence quality of the global Newton system, and it is often cumbersome and error-prone. 

To streamline this process, AD tools, such as those available in JAX, become highly valuable. These tools automate the differentiation process, reducing errors and enhancing the efficiency of computing the tangent operator. We will leverage JAX's AutoDiff capabilities extensively when working with `JAXMaterial` objects, see [](jax.md).

```{note}
For some generalized multi-physics behaviors, we may also need some derivatives of a state variable in $\mathcal{S}$ with respect to the imposed strain.
```

## Finite-strain setting

The above presentation which focused on a small strain setting can be directly extended towards the finite strain setting. We rely on a total Lagrangian formulation and write the equilibrium on the reference configuration which we still denote $\Omega$.

```{math}
:label: finite-strain-variational-model

\int_{\Omega} \bP(\bF):\nabla^s \bv \dOm = \int_\Omega \boldsymbol{f}\cdot\bv \dOm + \int_{\partial \Omega_\text{N}} \bT\cdot\bv \dS \quad \forall\bv \in V
```

where we use now the first Piola-Kirchhoff (PK1) stress $\bP(\bF)$ as an implicit function of the total deformation gradient $\bF=\bI+\nabla\bu$. Depending on the chosen material, some constitutive equations are more easily written by changing the stress/strain measure, using for instance the second Piola-Kirchhoff stress and the Green-Lagrange strain. Material libraries such as MFront typically provide some helper functions to convert the different stress/strain measures.

## Generalized behaviors

The same framework can also be extended to a multiphysics setting involving coupling of the mechanics with other physics such as heat or transport phenomena. To fix ideas, let us consider a thermo-mechanical system characterized by the heat and balance equations:

$$
\begin{align}
\int_{\Omega}\rho T_0 \dot{s}\widehat{T}d\Omega - \int_{\Omega} \boldsymbol{q}\cdot\nabla \widehat{T}d\Omega= 0 \quad \forall \widehat{T} \in V_T\\
\int_{\Omega} \bsig:\nabla^s \bv \dOm = \int_\Omega \boldsymbol{f}\cdot\bv \dOm + \int_{\partial \Omega_\text{N}} \bT\cdot\bv \dS \quad \forall\bv \in V_u
\end{align}
$$
where $T_0$ is a reference temperature, $s$ the entropy per unit of mass and $\boldsymbol{q}$ the heat flux.

In a fully coupled setting, $s,\bq$ and $\bsig$ are typically functions of the temperature $T$, the temperature gradient $\nabla T$ and the strain $\beps$. The same abstract constitutive mapping as in {eq}`constitutive-black-box` can therefore be considered by replacing the input with a set of *external state variables* (here the temperature $T$) and of *gradients* (here the temperature gradient $\nabla T$ and the strain $\beps$). The output of the constitutive equation is now a set of *fluxes* (here the heat flux $\bq$ and the stress $\bsig$) and of *internal state variables* (here the entropy). Again, to formulate a monolithic Newton method of this coupled system, various derivatives of outputs (fluxes + internal state variables) with respect to inputs (gradients and external state variables) should be provided by the material library. Finally, the heat and mechanics equations can also be solved in a staggered manner.

```{seealso}
For more details on such examples, we refer to the following MFront demos:
- [Nonlinear heat transfer](/demos/mfront/heat_transfer/nonlinear_heat_transfer.md)
- [Phase change](/demos/mfront/heat_transfer/phase_change.md)
```

## Computational aspects and Automatic Differentiation

The constitutive update process involves solving a small, local system of nonlinear evolution equations at each integration point to update the material state. This process, even for complex models like crystal plasticity with hundreds of state variables, is highly parallelizable and computationally efficient compared to solving the global system of equations governing the entire structure. Since the constitutive update takes up only a small fraction of the total computational time, there is flexibility to use more complex and expressive material models without significantly impacting overall performance. This context presents a promising opportunity for the application of automatic differentiation. AD can streamline the development and implementation of these complex constitutive models by automatically generating the necessary derivatives for the nonlinear equations and the associated consistent tangent operators, improving accuracy and reducing the manual effort required for coding, debugging, and optimizing these models. Consequently, modern AD tools can further enhance the expressiveness and ease of implementation of advanced material models without sacrificing computational efficiency.

However, attention should be paid as how to use AD in the context of constitutive modeling. For instance, performing AD on an unrolled version of the constitutive update program will not be efficient. The JAX section discusses various strategies, including custom differentiation using the implicit theorem to seamlessly perform AD on the constitutive update implementation.

(tensors_conventions)=
## Conventions for representing tensors

### 2nd-rank tensors

2nd-rank tensors are represented as vectors using the Mandel representation. Components ordering follow the conventions used by the MFront project [described here](https://thelfer.github.io/tfel/web/tensors.html).

A symmetric 2nd-rank tensor $\boldsymbol{a}$ stored as follows in 3D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{33} & \sqrt{2}a_{12} & \sqrt{2}a_{13} & \sqrt{2}a_{23} \end{Bmatrix}^\text{T}
$$
and in 2D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & \sqrt{2}a_{12} \end{Bmatrix}^\text{T}
$$

A non-symmetric 2nd-rank tensor $\boldsymbol{a}$ stored as follows in 3D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{33} & a_{12} & a_{21} & a_{13} & a_{31} & a_{23} & a_{32} \end{Bmatrix}^\text{T}
$$
and similarly in 2D:

$$
\{\boldsymbol{a}\} = \begin{Bmatrix}a_{11} & a_{22} & a_{12} & a_{21} \end{Bmatrix}^\text{T}
$$

### 4th-rank tensors

4th-rank tensors are represented as matrices with components complying with the representation of 2nd-rank tensors.

For instance, a symmetric 4th-order tensor in 2D will be represented as:

$$
[\mathbf{C}] &= \begin{bmatrix}
C_{1111} & C_{1122} & \sqrt{2}C_{1112} \\
C_{2211} & C_{2222} & \sqrt{2}C_{2212} \\
\sqrt{2}C_{1112} & \sqrt{2}C_{2212} & 2C_{1212}
\end{bmatrix}
$$

## References

```{bibliography}
:filter: docname in docnames
```
