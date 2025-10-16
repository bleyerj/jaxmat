# {py:mod}`jaxmat.solvers.custom_newton`

```{py:module} jaxmat.solvers.custom_newton
```

```{autodoc2-docstring} jaxmat.solvers.custom_newton
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`newton_solve_jittable <jaxmat.solvers.custom_newton.newton_solve_jittable>`
  - ```{autodoc2-docstring} jaxmat.solvers.custom_newton.newton_solve_jittable
    :summary:
    ```
````

### API

````{py:function} newton_solve_jittable(f: typing.Callable, x0: typing.Any, args: typing.Any, solver: lineax.AbstractLinearSolver, *, tol: float = 1e-06, maxiter: int = 20, damping: float = 1.0, has_aux: bool = False, jac='fwd')
:canonical: jaxmat.solvers.custom_newton.newton_solve_jittable

```{autodoc2-docstring} jaxmat.solvers.custom_newton.newton_solve_jittable
```
````
