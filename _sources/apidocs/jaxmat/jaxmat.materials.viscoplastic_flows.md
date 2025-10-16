# {py:mod}`jaxmat.materials.viscoplastic_flows`

```{py:module} jaxmat.materials.viscoplastic_flows
```

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VoceHardening <jaxmat.materials.viscoplastic_flows.VoceHardening>`
  -
* - {py:obj}`NortonFlow <jaxmat.materials.viscoplastic_flows.NortonFlow>`
  -
* - {py:obj}`AbstractKinematicHardening <jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening>`
  -
* - {py:obj}`ArmstrongFrederickHardening <jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening>`
  -
````

### API

`````{py:class} VoceHardening
:canonical: jaxmat.materials.viscoplastic_flows.VoceHardening

Bases: {py:obj}`equinox.Module`

````{py:attribute} sig0
:canonical: jaxmat.materials.viscoplastic_flows.VoceHardening.sig0
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.VoceHardening.sig0
```

````

````{py:attribute} sigu
:canonical: jaxmat.materials.viscoplastic_flows.VoceHardening.sigu
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.VoceHardening.sigu
```

````

````{py:attribute} b
:canonical: jaxmat.materials.viscoplastic_flows.VoceHardening.b
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.VoceHardening.b
```

````

````{py:method} __call__(p)
:canonical: jaxmat.materials.viscoplastic_flows.VoceHardening.__call__

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.VoceHardening.__call__
```

````

`````

`````{py:class} NortonFlow
:canonical: jaxmat.materials.viscoplastic_flows.NortonFlow

Bases: {py:obj}`equinox.Module`

````{py:attribute} K
:canonical: jaxmat.materials.viscoplastic_flows.NortonFlow.K
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.NortonFlow.K
```

````

````{py:attribute} m
:canonical: jaxmat.materials.viscoplastic_flows.NortonFlow.m
:type: float
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.NortonFlow.m
```

````

````{py:method} __call__(f)
:canonical: jaxmat.materials.viscoplastic_flows.NortonFlow.__call__

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.NortonFlow.__call__
```

````

`````

`````{py:class} AbstractKinematicHardening
:canonical: jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening

Bases: {py:obj}`equinox.Module`

````{py:attribute} nvars
:canonical: jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.nvars
:type: equinox.AbstractVar[int]
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.nvars
```

````

````{py:method} __call__(X, *args)
:canonical: jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.__call__
:abstractmethod:

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.__call__
```

````

````{py:method} sig_eff(sig, X)
:canonical: jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.sig_eff

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening.sig_eff
```

````

`````

`````{py:class} ArmstrongFrederickHardening
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening

Bases: {py:obj}`jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening`

````{py:attribute} C
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.C
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.C
```

````

````{py:attribute} g
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.g
:type: jax.Array
:value: >
   None

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.g
```

````

````{py:attribute} nvars
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.nvars
:value: >
   2

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.nvars
```

````

````{py:method} __call__(a, dp, depsp)
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.__call__

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.__call__
```

````

````{py:method} sig_eff(sig, a)
:canonical: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.sig_eff

```{autodoc2-docstring} jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening.sig_eff
```

````

`````
