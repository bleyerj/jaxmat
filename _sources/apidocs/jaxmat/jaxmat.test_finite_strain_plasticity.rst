:py:mod:`jaxmat.test_finite_strain_plasticity`
==============================================

.. py:module:: jaxmat.test_finite_strain_plasticity

.. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`YieldStress <jaxmat.test_finite_strain_plasticity.YieldStress>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`test_FeFp_elastoplasticity <jaxmat.test_finite_strain_plasticity.test_FeFp_elastoplasticity>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.test_FeFp_elastoplasticity
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`elastic_model <jaxmat.test_finite_strain_plasticity.elastic_model>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.elastic_model
          :summary:
   * - :py:obj:`sig0 <jaxmat.test_finite_strain_plasticity.sig0>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.sig0
          :summary:
   * - :py:obj:`sigu <jaxmat.test_finite_strain_plasticity.sigu>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.sigu
          :summary:
   * - :py:obj:`b <jaxmat.test_finite_strain_plasticity.b>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.b
          :summary:
   * - :py:obj:`Nbatch <jaxmat.test_finite_strain_plasticity.Nbatch>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.Nbatch
          :summary:
   * - :py:obj:`material <jaxmat.test_finite_strain_plasticity.material>`
     - .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.material
          :summary:

API
~~~

.. py:function:: test_FeFp_elastoplasticity(material, Nbatch=1)
   :canonical: jaxmat.test_finite_strain_plasticity.test_FeFp_elastoplasticity

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.test_FeFp_elastoplasticity

.. py:data:: elastic_model
   :canonical: jaxmat.test_finite_strain_plasticity.elastic_model
   :value: 'LinearElasticIsotropic(...)'

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.elastic_model

.. py:data:: sig0
   :canonical: jaxmat.test_finite_strain_plasticity.sig0
   :value: 350.0

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.sig0

.. py:data:: sigu
   :canonical: jaxmat.test_finite_strain_plasticity.sigu
   :value: 500.0

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.sigu

.. py:data:: b
   :canonical: jaxmat.test_finite_strain_plasticity.b
   :value: 1000.0

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.b

.. py:class:: YieldStress
   :canonical: jaxmat.test_finite_strain_plasticity.YieldStress

   Bases: :py:obj:`equinox.Module`

   .. py:method:: __call__(p)
      :canonical: jaxmat.test_finite_strain_plasticity.YieldStress.__call__

      .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.YieldStress.__call__

.. py:data:: Nbatch
   :canonical: jaxmat.test_finite_strain_plasticity.Nbatch
   :value: 'int(...)'

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.Nbatch

.. py:data:: material
   :canonical: jaxmat.test_finite_strain_plasticity.material
   :value: 'FeFpJ2Plasticity(...)'

   .. autodoc2-docstring:: jaxmat.test_finite_strain_plasticity.material
