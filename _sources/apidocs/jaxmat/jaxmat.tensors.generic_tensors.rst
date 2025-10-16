:py:mod:`jaxmat.tensors.generic_tensors`
========================================

.. py:module:: jaxmat.tensors.generic_tensors

.. autodoc2-docstring:: jaxmat.tensors.generic_tensors
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Tensor <jaxmat.tensors.generic_tensors.Tensor>`
     -
   * - :py:obj:`Tensor2 <jaxmat.tensors.generic_tensors.Tensor2>`
     -
   * - :py:obj:`SymmetricTensor2 <jaxmat.tensors.generic_tensors.SymmetricTensor2>`
     -
   * - :py:obj:`SymmetricTensor4 <jaxmat.tensors.generic_tensors.SymmetricTensor4>`
     -
   * - :py:obj:`IsotropicTensor4 <jaxmat.tensors.generic_tensors.IsotropicTensor4>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`symmetric_kelvin_mandel_index_map <jaxmat.tensors.generic_tensors.symmetric_kelvin_mandel_index_map>`
     - .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.symmetric_kelvin_mandel_index_map
          :summary:
   * - :py:obj:`_eval_basis <jaxmat.tensors.generic_tensors._eval_basis>`
     - .. autodoc2-docstring:: jaxmat.tensors.generic_tensors._eval_basis
          :summary:

API
~~~

.. py:class:: Tensor(tensor: typing.Optional[jax.Array] = None, array: typing.Optional[jax.Array] = None)
   :canonical: jaxmat.tensors.generic_tensors.Tensor

   Bases: :py:obj:`equinox.Module`

   .. py:attribute:: dim
      :canonical: jaxmat.tensors.generic_tensors.Tensor.dim
      :type: int
      :value: None

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.dim

   .. py:attribute:: rank
      :canonical: jaxmat.tensors.generic_tensors.Tensor.rank
      :type: int
      :value: None

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.rank

   .. py:attribute:: _tensor
      :canonical: jaxmat.tensors.generic_tensors.Tensor._tensor
      :type: jax.Array
      :value: None

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor._tensor

   .. py:property:: shape
      :canonical: jaxmat.tensors.generic_tensors.Tensor.shape

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.shape

   .. py:property:: tensor
      :canonical: jaxmat.tensors.generic_tensors.Tensor.tensor

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.tensor

   .. py:property:: T
      :canonical: jaxmat.tensors.generic_tensors.Tensor.T

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.T

   .. py:property:: array
      :canonical: jaxmat.tensors.generic_tensors.Tensor.array

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.array

   .. py:property:: array_shape
      :canonical: jaxmat.tensors.generic_tensors.Tensor.array_shape

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.array_shape

   .. py:method:: __getitem__(idx)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__getitem__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__getitem__

   .. py:method:: __jax_array__()
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__jax_array__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__jax_array__

   .. py:method:: __array__(dtype=None)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__array__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__array__

   .. py:method:: __add__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__add__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__add__

   .. py:method:: __sub__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__sub__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__sub__

   .. py:method:: __mul__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__mul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__mul__

   .. py:method:: __truediv__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__truediv__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__truediv__

   .. py:method:: __rmul__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__rmul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__rmul__

   .. py:method:: __matmul__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__matmul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__matmul__

   .. py:method:: __rmatmul__(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__rmatmul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__rmatmul__

   .. py:method:: __neg__()
      :canonical: jaxmat.tensors.generic_tensors.Tensor.__neg__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor.__neg__

   .. py:method:: _as_array(tensor)
      :canonical: jaxmat.tensors.generic_tensors.Tensor._as_array

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor._as_array

   .. py:method:: _as_tensor(array)
      :canonical: jaxmat.tensors.generic_tensors.Tensor._as_tensor

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor._as_tensor

   .. py:method:: _weaken_with(other)
      :canonical: jaxmat.tensors.generic_tensors.Tensor._weaken_with

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor._weaken_with

.. py:class:: Tensor2(tensor: typing.Optional[jax.Array] = None, array: typing.Optional[jax.Array] = None)
   :canonical: jaxmat.tensors.generic_tensors.Tensor2

   Bases: :py:obj:`jaxmat.tensors.generic_tensors.Tensor`

   .. py:attribute:: dim
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.dim
      :value: 3

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.dim

   .. py:attribute:: rank
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.rank
      :value: 2

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.rank

   .. py:method:: identity()
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.identity
      :classmethod:

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.identity

   .. py:method:: _as_array(tensor)
      :canonical: jaxmat.tensors.generic_tensors.Tensor2._as_array

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2._as_array

   .. py:method:: _as_tensor(array)
      :canonical: jaxmat.tensors.generic_tensors.Tensor2._as_tensor

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2._as_tensor

   .. py:property:: sym
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.sym

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.sym

   .. py:property:: inv
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.inv

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.inv

   .. py:property:: eigenvalues
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.eigenvalues

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.eigenvalues

   .. py:property:: T
      :canonical: jaxmat.tensors.generic_tensors.Tensor2.T

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.Tensor2.T

.. py:class:: SymmetricTensor2(tensor: typing.Optional[jax.Array] = None, array: typing.Optional[jax.Array] = None)
   :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2

   Bases: :py:obj:`jaxmat.tensors.generic_tensors.Tensor2`

   .. py:property:: array_shape
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2.array_shape

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2.array_shape

   .. py:method:: is_symmetric()
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2.is_symmetric

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2.is_symmetric

   .. py:method:: _as_array(tensor)
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2._as_array

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2._as_array

   .. py:method:: _as_tensor(array)
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2._as_tensor

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2._as_tensor

   .. py:method:: __matmul__(other)
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2.__matmul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2.__matmul__

   .. py:method:: _weaken_with(other)
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor2._weaken_with

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor2._weaken_with

.. py:function:: symmetric_kelvin_mandel_index_map(d)
   :canonical: jaxmat.tensors.generic_tensors.symmetric_kelvin_mandel_index_map

   .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.symmetric_kelvin_mandel_index_map

.. py:class:: SymmetricTensor4(tensor: typing.Optional[jax.Array] = None, array: typing.Optional[jax.Array] = None)
   :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4

   Bases: :py:obj:`jaxmat.tensors.generic_tensors.Tensor`

   .. py:attribute:: dim
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.dim
      :value: 3

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.dim

   .. py:attribute:: rank
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.rank
      :value: 4

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.rank

   .. py:method:: identity()
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.identity
      :classmethod:

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.identity

   .. py:method:: J()
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.J
      :classmethod:

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.J

   .. py:method:: K()
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.K
      :classmethod:

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.K

   .. py:property:: array_shape
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.array_shape

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.array_shape

   .. py:method:: is_symmetric()
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.is_symmetric

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.is_symmetric

   .. py:method:: _as_array(tensor: jax.Array) -> jax.Array
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4._as_array

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4._as_array

   .. py:method:: _as_tensor(array: jax.Array) -> jax.Array
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4._as_tensor

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4._as_tensor

   .. py:method:: __matmul__(other)
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.__matmul__

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.__matmul__

   .. py:property:: inv
      :canonical: jaxmat.tensors.generic_tensors.SymmetricTensor4.inv

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.SymmetricTensor4.inv

.. py:function:: _eval_basis(coeffs, basis)
   :canonical: jaxmat.tensors.generic_tensors._eval_basis

   .. autodoc2-docstring:: jaxmat.tensors.generic_tensors._eval_basis

.. py:class:: IsotropicTensor4(kappa, mu)
   :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4

   Bases: :py:obj:`jaxmat.tensors.generic_tensors.SymmetricTensor4`

   .. py:attribute:: kappa
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.kappa
      :type: float
      :value: None

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.kappa

   .. py:attribute:: mu
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.mu
      :type: float
      :value: None

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.mu

   .. py:property:: basis
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.basis

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.basis

   .. py:property:: coeffs
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.coeffs

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.coeffs

   .. py:method:: eval()
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.eval

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.eval

   .. py:property:: inv
      :canonical: jaxmat.tensors.generic_tensors.IsotropicTensor4.inv

      .. autodoc2-docstring:: jaxmat.tensors.generic_tensors.IsotropicTensor4.inv
