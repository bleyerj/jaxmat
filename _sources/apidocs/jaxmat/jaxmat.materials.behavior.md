# {py:mod}`jaxmat.materials.behavior`

```{py:module} jaxmat.materials.behavior
```

```{autodoc2-docstring} jaxmat.materials.behavior
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AbstractBehavior <jaxmat.materials.behavior.AbstractBehavior>`
  -
* - {py:obj}`SmallStrainBehavior <jaxmat.materials.behavior.SmallStrainBehavior>`
  -
* - {py:obj}`FiniteStrainBehavior <jaxmat.materials.behavior.FiniteStrainBehavior>`
  -
````

### API

`````{py:class} AbstractBehavior
:canonical: jaxmat.materials.behavior.AbstractBehavior

Bases: {py:obj}`equinox.Module`

````{py:attribute} internal
:canonical: jaxmat.materials.behavior.AbstractBehavior.internal
:type: equinox.AbstractVar[jaxmat.state.AbstractState]
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior.internal
```

````

````{py:attribute} solver
:canonical: jaxmat.materials.behavior.AbstractBehavior.solver
:type: optimistix.AbstractRootFinder
:value: >
   'field(...)'

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior.solver
```

````

````{py:attribute} adjoint
:canonical: jaxmat.materials.behavior.AbstractBehavior.adjoint
:type: optimistix.AbstractAdjoint
:value: >
   'field(...)'

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior.adjoint
```

````

````{py:attribute} _batch_size
:canonical: jaxmat.materials.behavior.AbstractBehavior._batch_size
:type: jax.Array
:value: >
   'field(...)'

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior._batch_size
```

````

````{py:method} constitutive_update(eps, state, dt)
:canonical: jaxmat.materials.behavior.AbstractBehavior.constitutive_update
:abstractmethod:

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior.constitutive_update
```

````

````{py:method} batched_constitutive_update(eps, state, dt)
:canonical: jaxmat.materials.behavior.AbstractBehavior.batched_constitutive_update

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior.batched_constitutive_update
```

````

````{py:method} _init_state(cls, Nbatch=None)
:canonical: jaxmat.materials.behavior.AbstractBehavior._init_state

```{autodoc2-docstring} jaxmat.materials.behavior.AbstractBehavior._init_state
```

````

`````

`````{py:class} SmallStrainBehavior
:canonical: jaxmat.materials.behavior.SmallStrainBehavior

Bases: {py:obj}`jaxmat.materials.behavior.AbstractBehavior`

````{py:method} init_state(Nbatch=None)
:canonical: jaxmat.materials.behavior.SmallStrainBehavior.init_state

```{autodoc2-docstring} jaxmat.materials.behavior.SmallStrainBehavior.init_state
```

````

`````

`````{py:class} FiniteStrainBehavior
:canonical: jaxmat.materials.behavior.FiniteStrainBehavior

Bases: {py:obj}`jaxmat.materials.behavior.AbstractBehavior`

````{py:method} init_state(Nbatch=None)
:canonical: jaxmat.materials.behavior.FiniteStrainBehavior.init_state

```{autodoc2-docstring} jaxmat.materials.behavior.FiniteStrainBehavior.init_state
```

````

`````
