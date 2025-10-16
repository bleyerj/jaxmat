:py:mod:`jaxmat.test_elastoplasticity`
======================================

.. py:module:: jaxmat.test_elastoplasticity

.. autodoc2-docstring:: jaxmat.test_elastoplasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`YieldStress <jaxmat.test_elastoplasticity.YieldStress>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`test_elastoplasticity <jaxmat.test_elastoplasticity.test_elastoplasticity>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.test_elastoplasticity
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`elastic_model <jaxmat.test_elastoplasticity.elastic_model>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.elastic_model
          :summary:
   * - :py:obj:`sig0 <jaxmat.test_elastoplasticity.sig0>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.sig0
          :summary:
   * - :py:obj:`sigu <jaxmat.test_elastoplasticity.sigu>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.sigu
          :summary:
   * - :py:obj:`b <jaxmat.test_elastoplasticity.b>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.b
          :summary:
   * - :py:obj:`Nbatch <jaxmat.test_elastoplasticity.Nbatch>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.Nbatch
          :summary:
   * - :py:obj:`material <jaxmat.test_elastoplasticity.material>`
     - .. autodoc2-docstring:: jaxmat.test_elastoplasticity.material
          :summary:

API
~~~

.. py:function:: test_elastoplasticity(material, Nbatch=1)
   :canonical: jaxmat.test_elastoplasticity.test_elastoplasticity

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.test_elastoplasticity

.. py:data:: elastic_model
   :canonical: jaxmat.test_elastoplasticity.elastic_model
   :value: 'LinearElasticIsotropic(...)'

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.elastic_model

.. py:data:: sig0
   :canonical: jaxmat.test_elastoplasticity.sig0
   :value: 350.0

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.sig0

.. py:data:: sigu
   :canonical: jaxmat.test_elastoplasticity.sigu
   :value: 500.0

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.sigu

.. py:data:: b
   :canonical: jaxmat.test_elastoplasticity.b
   :value: 1000.0

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.b

.. py:class:: YieldStress
   :canonical: jaxmat.test_elastoplasticity.YieldStress

   Bases: :py:obj:`equinox.Module`

   .. py:method:: __call__(p)
      :canonical: jaxmat.test_elastoplasticity.YieldStress.__call__

      .. autodoc2-docstring:: jaxmat.test_elastoplasticity.YieldStress.__call__

.. py:data:: Nbatch
   :canonical: jaxmat.test_elastoplasticity.Nbatch
   :value: 'int(...)'

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.Nbatch

.. py:data:: material
   :canonical: jaxmat.test_elastoplasticity.material
   :value: 'vonMisesIsotropicHardening(...)'

   .. autodoc2-docstring:: jaxmat.test_elastoplasticity.material
