# {py:mod}`jaxmat.materials.hyperelasticity`

```{py:module} jaxmat.materials.hyperelasticity
```

```{autodoc2-docstring} jaxmat.materials.hyperelasticity
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HyperelasticPotential <jaxmat.materials.hyperelasticity.HyperelasticPotential>`
  -
* - {py:obj}`Hyperelasticity <jaxmat.materials.hyperelasticity.Hyperelasticity>`
  -
* - {py:obj}`VolumetricPart <jaxmat.materials.hyperelasticity.VolumetricPart>`
  -
* - {py:obj}`SquaredVolumetric <jaxmat.materials.hyperelasticity.SquaredVolumetric>`
  -
* - {py:obj}`CompressibleNeoHookean <jaxmat.materials.hyperelasticity.CompressibleNeoHookean>`
  -
* - {py:obj}`CompressibleMooneyRivlin <jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin>`
  -
* - {py:obj}`CompressibleGhentMooneyRivlin <jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin>`
  -
* - {py:obj}`CompressibleOgden <jaxmat.materials.hyperelasticity.CompressibleOgden>`
  -
````

### API

`````{py:class} HyperelasticPotential
:canonical: jaxmat.materials.hyperelasticity.HyperelasticPotential

Bases: {py:obj}`equinox.Module`

````{py:method} __call__()
:canonical: jaxmat.materials.hyperelasticity.HyperelasticPotential.__call__
:abstractmethod:

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.HyperelasticPotential.__call__
```

````

````{py:method} PK1(F)
:canonical: jaxmat.materials.hyperelasticity.HyperelasticPotential.PK1

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.HyperelasticPotential.PK1
```

````

````{py:method} PK2(F)
:canonical: jaxmat.materials.hyperelasticity.HyperelasticPotential.PK2

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.HyperelasticPotential.PK2
```

````

````{py:method} Cauchy(F)
:canonical: jaxmat.materials.hyperelasticity.HyperelasticPotential.Cauchy

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.HyperelasticPotential.Cauchy
```

````

`````

`````{py:class} Hyperelasticity
:canonical: jaxmat.materials.hyperelasticity.Hyperelasticity

Bases: {py:obj}`jaxmat.materials.behavior.FiniteStrainBehavior`

````{py:attribute} potential
:canonical: jaxmat.materials.hyperelasticity.Hyperelasticity.potential
:type: jaxmat.materials.hyperelasticity.HyperelasticPotential
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.Hyperelasticity.potential
```

````

````{py:attribute} internal
:canonical: jaxmat.materials.hyperelasticity.Hyperelasticity.internal
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.Hyperelasticity.internal
```

````

````{py:method} constitutive_update(F, state, dt)
:canonical: jaxmat.materials.hyperelasticity.Hyperelasticity.constitutive_update

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.Hyperelasticity.constitutive_update
```

````

`````

`````{py:class} VolumetricPart
:canonical: jaxmat.materials.hyperelasticity.VolumetricPart

Bases: {py:obj}`equinox.Module`

````{py:attribute} beta
:canonical: jaxmat.materials.hyperelasticity.VolumetricPart.beta
:type: float
:value: >
   2.0

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.VolumetricPart.beta
```

````

````{py:method} __call__(J)
:canonical: jaxmat.materials.hyperelasticity.VolumetricPart.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.VolumetricPart.__call__
```

````

`````

`````{py:class} SquaredVolumetric
:canonical: jaxmat.materials.hyperelasticity.SquaredVolumetric

Bases: {py:obj}`equinox.Module`

````{py:method} __call__(J)
:canonical: jaxmat.materials.hyperelasticity.SquaredVolumetric.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.SquaredVolumetric.__call__
```

````

`````

`````{py:class} CompressibleNeoHookean
:canonical: jaxmat.materials.hyperelasticity.CompressibleNeoHookean

Bases: {py:obj}`jaxmat.materials.hyperelasticity.HyperelasticPotential`

````{py:attribute} mu
:canonical: jaxmat.materials.hyperelasticity.CompressibleNeoHookean.mu
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleNeoHookean.mu
```

````

````{py:attribute} kappa
:canonical: jaxmat.materials.hyperelasticity.CompressibleNeoHookean.kappa
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleNeoHookean.kappa
```

````

````{py:attribute} volumetric
:canonical: jaxmat.materials.hyperelasticity.CompressibleNeoHookean.volumetric
:type: equinox.Module
:value: >
   'VolumetricPart(...)'

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleNeoHookean.volumetric
```

````

````{py:method} __call__(F)
:canonical: jaxmat.materials.hyperelasticity.CompressibleNeoHookean.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleNeoHookean.__call__
```

````

`````

`````{py:class} CompressibleMooneyRivlin
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin

Bases: {py:obj}`jaxmat.materials.hyperelasticity.HyperelasticPotential`

````{py:attribute} c1
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.c1
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.c1
```

````

````{py:attribute} c2
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.c2
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.c2
```

````

````{py:attribute} kappa
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.kappa
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.kappa
```

````

````{py:attribute} volumetric
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.volumetric
:type: equinox.Module
:value: >
   'VolumetricPart(...)'

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.volumetric
```

````

````{py:method} __call__(F)
:canonical: jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleMooneyRivlin.__call__
```

````

`````

`````{py:class} CompressibleGhentMooneyRivlin
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin

Bases: {py:obj}`jaxmat.materials.hyperelasticity.HyperelasticPotential`

````{py:attribute} c1
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.c1
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.c1
```

````

````{py:attribute} c2
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.c2
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.c2
```

````

````{py:attribute} Jm
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.Jm
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.Jm
```

````

````{py:attribute} kappa
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.kappa
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.kappa
```

````

````{py:attribute} volumetric
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.volumetric
:type: equinox.Module
:value: >
   'VolumetricPart(...)'

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.volumetric
```

````

````{py:method} __call__(F)
:canonical: jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleGhentMooneyRivlin.__call__
```

````

`````

`````{py:class} CompressibleOgden
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden

Bases: {py:obj}`jaxmat.materials.hyperelasticity.HyperelasticPotential`

````{py:attribute} mu
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.mu
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.mu
```

````

````{py:attribute} alpha
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.alpha
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.alpha
```

````

````{py:attribute} kappa
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.kappa
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.kappa
```

````

````{py:attribute} volumetric
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.volumetric
:type: equinox.Module
:value: >
   'VolumetricPart(...)'

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.volumetric
```

````

````{py:method} __call__(F)
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.__call__

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.__call__
```

````

````{py:method} W_lamb(lambCb)
:canonical: jaxmat.materials.hyperelasticity.CompressibleOgden.W_lamb

```{autodoc2-docstring} jaxmat.materials.hyperelasticity.CompressibleOgden.W_lamb
```

````

`````
