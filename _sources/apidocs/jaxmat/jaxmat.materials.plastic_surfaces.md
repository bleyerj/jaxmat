# {py:mod}`jaxmat.materials.plastic_surfaces`

```{py:module} jaxmat.materials.plastic_surfaces
```

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AbstractPlasticSurface <jaxmat.materials.plastic_surfaces.AbstractPlasticSurface>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
    :summary:
    ```
* - {py:obj}`vonMises <jaxmat.materials.plastic_surfaces.vonMises>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.vonMises
    :summary:
    ```
* - {py:obj}`DruckerPrager <jaxmat.materials.plastic_surfaces.DruckerPrager>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.DruckerPrager
    :summary:
    ```
* - {py:obj}`Hosford <jaxmat.materials.plastic_surfaces.Hosford>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.Hosford
    :summary:
    ```
* - {py:obj}`Tresca <jaxmat.materials.plastic_surfaces.Tresca>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.Tresca
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`safe_zero <jaxmat.materials.plastic_surfaces.safe_zero>`
  - ```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.safe_zero
    :summary:
    ```
````

### API

````{py:function} safe_zero(method)
:canonical: jaxmat.materials.plastic_surfaces.safe_zero

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.safe_zero
```
````

`````{py:class} AbstractPlasticSurface
:canonical: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface

Bases: {py:obj}`equinox.Module`

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
```

````{py:method} __call__(sig, *args)
:canonical: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface.__call__
:abstractmethod:

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.AbstractPlasticSurface.__call__
```

````

````{py:method} normal(sig, *args)
:canonical: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface.normal

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.AbstractPlasticSurface.normal
```

````

`````

`````{py:class} vonMises
:canonical: jaxmat.materials.plastic_surfaces.vonMises

Bases: {py:obj}`jaxmat.materials.plastic_surfaces.AbstractPlasticSurface`

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.vonMises
```

````{py:method} __call__(sig)
:canonical: jaxmat.materials.plastic_surfaces.vonMises.__call__

````

`````

`````{py:class} DruckerPrager
:canonical: jaxmat.materials.plastic_surfaces.DruckerPrager

Bases: {py:obj}`jaxmat.materials.plastic_surfaces.AbstractPlasticSurface`

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.DruckerPrager
```

````{py:attribute} alpha
:canonical: jaxmat.materials.plastic_surfaces.DruckerPrager.alpha
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.DruckerPrager.alpha
```

````

````{py:method} __call__(sig)
:canonical: jaxmat.materials.plastic_surfaces.DruckerPrager.__call__

````

`````

`````{py:class} Hosford
:canonical: jaxmat.materials.plastic_surfaces.Hosford

Bases: {py:obj}`jaxmat.materials.plastic_surfaces.AbstractPlasticSurface`

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.Hosford
```

````{py:attribute} a
:canonical: jaxmat.materials.plastic_surfaces.Hosford.a
:type: float
:value: >
   'field(...)'

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.Hosford.a
```

````

````{py:method} __call__(sig)
:canonical: jaxmat.materials.plastic_surfaces.Hosford.__call__

````

`````

`````{py:class} Tresca
:canonical: jaxmat.materials.plastic_surfaces.Tresca

Bases: {py:obj}`jaxmat.materials.plastic_surfaces.AbstractPlasticSurface`

```{autodoc2-docstring} jaxmat.materials.plastic_surfaces.Tresca
```

````{py:method} __call__(sig)
:canonical: jaxmat.materials.plastic_surfaces.Tresca.__call__

````

`````
