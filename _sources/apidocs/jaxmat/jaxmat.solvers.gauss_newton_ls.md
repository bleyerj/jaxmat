# {py:mod}`jaxmat.solvers.gauss_newton_ls`

```{py:module} jaxmat.solvers.gauss_newton_ls
```

```{autodoc2-docstring} jaxmat.solvers.gauss_newton_ls
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GaussNewtonLineSearch <jaxmat.solvers.gauss_newton_ls.GaussNewtonLineSearch>`
  - ```{autodoc2-docstring} jaxmat.solvers.gauss_newton_ls.GaussNewtonLineSearch
    :summary:
    ```
````

### API

````{py:class} GaussNewtonLineSearch(rtol: float, atol: float, norm: collections.abc.Callable[[jaxtyping.PyTree], jaxtyping.Scalar] = max_norm, linear_solver: lineax.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None), verbose: frozenset[str] = frozenset())
:canonical: jaxmat.solvers.gauss_newton_ls.GaussNewtonLineSearch

Bases: {py:obj}`optimistix.GaussNewton`

```{autodoc2-docstring} jaxmat.solvers.gauss_newton_ls.GaussNewtonLineSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} jaxmat.solvers.gauss_newton_ls.GaussNewtonLineSearch.__init__
```

````
