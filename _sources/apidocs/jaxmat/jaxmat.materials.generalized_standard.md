# {py:mod}`jaxmat.materials.generalized_standard`

```{py:module} jaxmat.materials.generalized_standard
```

```{autodoc2-docstring} jaxmat.materials.generalized_standard
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InternalState <jaxmat.materials.generalized_standard.InternalState>`
  -
* - {py:obj}`FreeEnergy <jaxmat.materials.generalized_standard.FreeEnergy>`
  -
* - {py:obj}`DissipationPotential <jaxmat.materials.generalized_standard.DissipationPotential>`
  -
* - {py:obj}`MyState <jaxmat.materials.generalized_standard.MyState>`
  -
* - {py:obj}`GeneralizedStandardMaterial <jaxmat.materials.generalized_standard.GeneralizedStandardMaterial>`
  -
````

### API

`````{py:class} InternalState
:canonical: jaxmat.materials.generalized_standard.InternalState

Bases: {py:obj}`jaxmat.state.AbstractState`

````{py:attribute} epsv
:canonical: jaxmat.materials.generalized_standard.InternalState.epsv
:type: jaxmat.tensors.SymmetricTensor2
:value: >
   'SymmetricTensor2(...)'

```{autodoc2-docstring} jaxmat.materials.generalized_standard.InternalState.epsv
```

````

`````

`````{py:class} FreeEnergy
:canonical: jaxmat.materials.generalized_standard.FreeEnergy

Bases: {py:obj}`equinox.Module`

````{py:attribute} elastic_model
:canonical: jaxmat.materials.generalized_standard.FreeEnergy.elastic_model
:type: jaxmat.materials.elasticity.LinearElasticIsotropic
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.FreeEnergy.elastic_model
```

````

````{py:attribute} viscous_model
:canonical: jaxmat.materials.generalized_standard.FreeEnergy.viscous_model
:type: jaxmat.materials.elasticity.LinearElasticIsotropic
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.FreeEnergy.viscous_model
```

````

````{py:method} __call__(eps, isv)
:canonical: jaxmat.materials.generalized_standard.FreeEnergy.__call__

```{autodoc2-docstring} jaxmat.materials.generalized_standard.FreeEnergy.__call__
```

````

`````

`````{py:class} DissipationPotential
:canonical: jaxmat.materials.generalized_standard.DissipationPotential

Bases: {py:obj}`equinox.Module`

````{py:attribute} eta
:canonical: jaxmat.materials.generalized_standard.DissipationPotential.eta
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.DissipationPotential.eta
```

````

````{py:method} __call__(isv_dot)
:canonical: jaxmat.materials.generalized_standard.DissipationPotential.__call__

```{autodoc2-docstring} jaxmat.materials.generalized_standard.DissipationPotential.__call__
```

````

`````

`````{py:class} MyState
:canonical: jaxmat.materials.generalized_standard.MyState

Bases: {py:obj}`jaxmat.state.SmallStrainState`

````{py:attribute} internal_old
:canonical: jaxmat.materials.generalized_standard.MyState.internal_old
:type: jaxmat.state.AbstractState
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.MyState.internal_old
```

````

````{py:attribute} internal
:canonical: jaxmat.materials.generalized_standard.MyState.internal
:type: jaxmat.state.AbstractState
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.MyState.internal
```

````

`````

`````{py:class} GeneralizedStandardMaterial
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial

Bases: {py:obj}`jaxmat.materials.behavior.SmallStrainBehavior`

````{py:attribute} free_energy
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.free_energy
:type: equinox.Module
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.free_energy
```

````

````{py:attribute} dissipation_potential
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.dissipation_potential
:type: equinox.Module
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.dissipation_potential
```

````

````{py:attribute} internal
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.internal
:value: >
   'InternalState(...)'

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.internal
```

````

````{py:attribute} internal_old
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.internal_old
:value: >
   'InternalState(...)'

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.internal_old
```

````

````{py:attribute} solver
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.solver
:value: >
   'BFGS(...)'

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.solver
```

````

````{py:method} init_state(Nbatch)
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.init_state

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.init_state
```

````

````{py:method} incremental_potential(d_isv, args)
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.incremental_potential

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.incremental_potential
```

````

````{py:method} constitutive_update(eps, state, dt)
:canonical: jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.constitutive_update

```{autodoc2-docstring} jaxmat.materials.generalized_standard.GeneralizedStandardMaterial.constitutive_update
```

````

`````
