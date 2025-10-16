# {py:mod}`jaxmat.state`

```{py:module} jaxmat.state
```

```{autodoc2-docstring} jaxmat.state
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AbstractState <jaxmat.state.AbstractState>`
  -
* - {py:obj}`SmallStrainState <jaxmat.state.SmallStrainState>`
  -
* - {py:obj}`FiniteStrainState <jaxmat.state.FiniteStrainState>`
  -
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PK1_to_PK2 <jaxmat.state.PK1_to_PK2>`
  - ```{autodoc2-docstring} jaxmat.state.PK1_to_PK2
    :summary:
    ```
* - {py:obj}`PK1_to_Cauchy <jaxmat.state.PK1_to_Cauchy>`
  - ```{autodoc2-docstring} jaxmat.state.PK1_to_Cauchy
    :summary:
    ```
* - {py:obj}`make_batched <jaxmat.state.make_batched>`
  - ```{autodoc2-docstring} jaxmat.state.make_batched
    :summary:
    ```
````

### API

`````{py:class} AbstractState
:canonical: jaxmat.state.AbstractState

Bases: {py:obj}`equinox.Module`

````{py:method} _resolve_aliases(changes)
:canonical: jaxmat.state.AbstractState._resolve_aliases

```{autodoc2-docstring} jaxmat.state.AbstractState._resolve_aliases
```

````

````{py:method} add(**changes)
:canonical: jaxmat.state.AbstractState.add

```{autodoc2-docstring} jaxmat.state.AbstractState.add
```

````

````{py:method} update(**changes)
:canonical: jaxmat.state.AbstractState.update

```{autodoc2-docstring} jaxmat.state.AbstractState.update
```

````

`````

`````{py:class} SmallStrainState
:canonical: jaxmat.state.SmallStrainState

Bases: {py:obj}`jaxmat.state.AbstractState`

````{py:attribute} internal
:canonical: jaxmat.state.SmallStrainState.internal
:type: jaxmat.state.AbstractState
:value: >
   None

```{autodoc2-docstring} jaxmat.state.SmallStrainState.internal
```

````

````{py:attribute} strain
:canonical: jaxmat.state.SmallStrainState.strain
:type: jaxmat.tensors.SymmetricTensor2
:value: >
   'SymmetricTensor2(...)'

```{autodoc2-docstring} jaxmat.state.SmallStrainState.strain
```

````

````{py:attribute} stress
:canonical: jaxmat.state.SmallStrainState.stress
:type: jaxmat.tensors.SymmetricTensor2
:value: >
   'SymmetricTensor2(...)'

```{autodoc2-docstring} jaxmat.state.SmallStrainState.stress
```

````

````{py:attribute} __alias_targets__
:canonical: jaxmat.state.SmallStrainState.__alias_targets__
:value: >
   None

```{autodoc2-docstring} jaxmat.state.SmallStrainState.__alias_targets__
```

````

````{py:property} eps
:canonical: jaxmat.state.SmallStrainState.eps

```{autodoc2-docstring} jaxmat.state.SmallStrainState.eps
```

````

````{py:property} sig
:canonical: jaxmat.state.SmallStrainState.sig

```{autodoc2-docstring} jaxmat.state.SmallStrainState.sig
```

````

`````

````{py:function} PK1_to_PK2(F, PK1)
:canonical: jaxmat.state.PK1_to_PK2

```{autodoc2-docstring} jaxmat.state.PK1_to_PK2
```
````

````{py:function} PK1_to_Cauchy(F, PK1)
:canonical: jaxmat.state.PK1_to_Cauchy

```{autodoc2-docstring} jaxmat.state.PK1_to_Cauchy
```
````

`````{py:class} FiniteStrainState
:canonical: jaxmat.state.FiniteStrainState

Bases: {py:obj}`jaxmat.state.AbstractState`

````{py:attribute} internal
:canonical: jaxmat.state.FiniteStrainState.internal
:type: jaxmat.state.AbstractState
:value: >
   None

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.internal
```

````

````{py:attribute} strain
:canonical: jaxmat.state.FiniteStrainState.strain
:type: jaxmat.tensors.Tensor2
:value: >
   'identity(...)'

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.strain
```

````

````{py:attribute} stress
:canonical: jaxmat.state.FiniteStrainState.stress
:type: jaxmat.tensors.Tensor2
:value: >
   'Tensor2(...)'

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.stress
```

````

````{py:attribute} __alias_targets__
:canonical: jaxmat.state.FiniteStrainState.__alias_targets__
:value: >
   None

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.__alias_targets__
```

````

````{py:property} F
:canonical: jaxmat.state.FiniteStrainState.F

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.F
```

````

````{py:property} PK1
:canonical: jaxmat.state.FiniteStrainState.PK1

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.PK1
```

````

````{py:property} PK2
:canonical: jaxmat.state.FiniteStrainState.PK2

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.PK2
```

````

````{py:property} sig
:canonical: jaxmat.state.FiniteStrainState.sig

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.sig
```

````

````{py:property} Cauchy
:canonical: jaxmat.state.FiniteStrainState.Cauchy

```{autodoc2-docstring} jaxmat.state.FiniteStrainState.Cauchy
```

````

`````

````{py:function} make_batched(module: equinox.Module, Nbatch: int) -> equinox.Module
:canonical: jaxmat.state.make_batched

```{autodoc2-docstring} jaxmat.state.make_batched
```
````
