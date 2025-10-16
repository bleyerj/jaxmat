:py:mod:`jaxmat.tensors.linear_algebra`
=======================================

.. py:module:: jaxmat.tensors.linear_algebra

.. autodoc2-docstring:: jaxmat.tensors.linear_algebra
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`dim <jaxmat.tensors.linear_algebra.dim>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.dim
          :summary:
   * - :py:obj:`tr <jaxmat.tensors.linear_algebra.tr>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.tr
          :summary:
   * - :py:obj:`dev <jaxmat.tensors.linear_algebra.dev>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.dev
          :summary:
   * - :py:obj:`det33 <jaxmat.tensors.linear_algebra.det33>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.det33
          :summary:
   * - :py:obj:`inv33 <jaxmat.tensors.linear_algebra.inv33>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.inv33
          :summary:
   * - :py:obj:`invariants_principal <jaxmat.tensors.linear_algebra.invariants_principal>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.invariants_principal
          :summary:
   * - :py:obj:`invariants_main <jaxmat.tensors.linear_algebra.invariants_main>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.invariants_main
          :summary:
   * - :py:obj:`pq_invariants <jaxmat.tensors.linear_algebra.pq_invariants>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.pq_invariants
          :summary:
   * - :py:obj:`eig33 <jaxmat.tensors.linear_algebra.eig33>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.eig33
          :summary:
   * - :py:obj:`_sqrtm <jaxmat.tensors.linear_algebra._sqrtm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra._sqrtm
          :summary:
   * - :py:obj:`sqrtm <jaxmat.tensors.linear_algebra.sqrtm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.sqrtm
          :summary:
   * - :py:obj:`inv_sqrtm <jaxmat.tensors.linear_algebra.inv_sqrtm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.inv_sqrtm
          :summary:
   * - :py:obj:`isotropic_function <jaxmat.tensors.linear_algebra.isotropic_function>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.isotropic_function
          :summary:
   * - :py:obj:`expm <jaxmat.tensors.linear_algebra.expm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.expm
          :summary:
   * - :py:obj:`logm <jaxmat.tensors.linear_algebra.logm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.logm
          :summary:
   * - :py:obj:`powm <jaxmat.tensors.linear_algebra.powm>`
     - .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.powm
          :summary:

API
~~~

.. py:function:: dim(A)
   :canonical: jaxmat.tensors.linear_algebra.dim

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.dim

.. py:function:: tr(A)
   :canonical: jaxmat.tensors.linear_algebra.tr

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.tr

.. py:function:: dev(A)
   :canonical: jaxmat.tensors.linear_algebra.dev

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.dev

.. py:function:: det33(A)
   :canonical: jaxmat.tensors.linear_algebra.det33

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.det33

.. py:function:: inv33(A)
   :canonical: jaxmat.tensors.linear_algebra.inv33

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.inv33

.. py:function:: invariants_principal(A)
   :canonical: jaxmat.tensors.linear_algebra.invariants_principal

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.invariants_principal

.. py:function:: invariants_main(A)
   :canonical: jaxmat.tensors.linear_algebra.invariants_main

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.invariants_main

.. py:function:: pq_invariants(sig)
   :canonical: jaxmat.tensors.linear_algebra.pq_invariants

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.pq_invariants

.. py:function:: eig33(A, rtol=1e-16)
   :canonical: jaxmat.tensors.linear_algebra.eig33

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.eig33

.. py:function:: _sqrtm(C)
   :canonical: jaxmat.tensors.linear_algebra._sqrtm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra._sqrtm

.. py:function:: sqrtm(A)
   :canonical: jaxmat.tensors.linear_algebra.sqrtm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.sqrtm

.. py:function:: inv_sqrtm(A)
   :canonical: jaxmat.tensors.linear_algebra.inv_sqrtm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.inv_sqrtm

.. py:function:: isotropic_function(fun, A)
   :canonical: jaxmat.tensors.linear_algebra.isotropic_function

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.isotropic_function

.. py:function:: expm(A)
   :canonical: jaxmat.tensors.linear_algebra.expm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.expm

.. py:function:: logm(A)
   :canonical: jaxmat.tensors.linear_algebra.logm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.logm

.. py:function:: powm(A, m)
   :canonical: jaxmat.tensors.linear_algebra.powm

   .. autodoc2-docstring:: jaxmat.tensors.linear_algebra.powm
