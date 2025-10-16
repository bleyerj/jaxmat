# {py:mod}`jaxmat.loader`

```{py:module} jaxmat.loader
```

```{autodoc2-docstring} jaxmat.loader
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImposedLoading <jaxmat.loader.ImposedLoading>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_make_imposed_loading <jaxmat.loader._make_imposed_loading>`
  - ```{autodoc2-docstring} jaxmat.loader._make_imposed_loading
    :summary:
    ```
* - {py:obj}`residual <jaxmat.loader.residual>`
  - ```{autodoc2-docstring} jaxmat.loader.residual
    :summary:
    ```
* - {py:obj}`stack_loadings <jaxmat.loader.stack_loadings>`
  - ```{autodoc2-docstring} jaxmat.loader.stack_loadings
    :summary:
    ```
* - {py:obj}`solve_mechanical_state <jaxmat.loader.solve_mechanical_state>`
  - ```{autodoc2-docstring} jaxmat.loader.solve_mechanical_state
    :summary:
    ```
* - {py:obj}`global_solve <jaxmat.loader.global_solve>`
  - ```{autodoc2-docstring} jaxmat.loader.global_solve
    :summary:
    ```
````

### API

`````{py:class} ImposedLoading(hypothesis: typing.Literal[small_strain, finite_strain] = 'small_strain', **kwargs)
:canonical: jaxmat.loader.ImposedLoading

Bases: {py:obj}`equinox.Module`

````{py:attribute} eps_vals
:canonical: jaxmat.loader.ImposedLoading.eps_vals
:type: jax.numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jaxmat.loader.ImposedLoading.eps_vals
```

````

````{py:attribute} sig_vals
:canonical: jaxmat.loader.ImposedLoading.sig_vals
:type: jax.numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jaxmat.loader.ImposedLoading.sig_vals
```

````

````{py:attribute} strain_mask
:canonical: jaxmat.loader.ImposedLoading.strain_mask
:type: jax.numpy.ndarray
:value: >
   None

```{autodoc2-docstring} jaxmat.loader.ImposedLoading.strain_mask
```

````

````{py:method} __call__()
:canonical: jaxmat.loader.ImposedLoading.__call__

```{autodoc2-docstring} jaxmat.loader.ImposedLoading.__call__
```

````

````{py:method} __len__()
:canonical: jaxmat.loader.ImposedLoading.__len__

```{autodoc2-docstring} jaxmat.loader.ImposedLoading.__len__
```

````

`````

````{py:function} _make_imposed_loading(hypothesis: typing.Literal[small_strain, finite_strain] = 'small_strain', **kwargs) -> jaxmat.loader.ImposedLoading
:canonical: jaxmat.loader._make_imposed_loading

```{autodoc2-docstring} jaxmat.loader._make_imposed_loading
```
````

````{py:function} residual(material, loader: jaxmat.loader.ImposedLoading, eps: jax.numpy.ndarray, state: dict, dt: float)
:canonical: jaxmat.loader.residual

```{autodoc2-docstring} jaxmat.loader.residual
```
````

````{py:function} stack_loadings(loadings: list)
:canonical: jaxmat.loader.stack_loadings

```{autodoc2-docstring} jaxmat.loader.stack_loadings
```
````

````{py:function} solve_mechanical_state(eps0, state, loading_data: jaxmat.loader.ImposedLoading, material, dt)
:canonical: jaxmat.loader.solve_mechanical_state

```{autodoc2-docstring} jaxmat.loader.solve_mechanical_state
```
````

````{py:function} global_solve(Eps0, state, loading_data, material, dt, in_axes=(0, 0, 0, None, None))
:canonical: jaxmat.loader.global_solve

```{autodoc2-docstring} jaxmat.loader.global_solve
```
````
