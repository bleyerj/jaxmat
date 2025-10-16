:py:mod:`jaxmat.test_gsm`
=========================

.. py:module:: jaxmat.test_gsm

.. autodoc2-docstring:: jaxmat.test_gsm
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`test_elastoplasticity <jaxmat.test_gsm.test_elastoplasticity>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.test_elastoplasticity
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`elastic_model <jaxmat.test_gsm.elastic_model>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.elastic_model
          :summary:
   * - :py:obj:`viscous_model <jaxmat.test_gsm.viscous_model>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.viscous_model
          :summary:
   * - :py:obj:`Nbatch <jaxmat.test_gsm.Nbatch>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.Nbatch
          :summary:
   * - :py:obj:`free_energy <jaxmat.test_gsm.free_energy>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.free_energy
          :summary:
   * - :py:obj:`dissipation <jaxmat.test_gsm.dissipation>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.dissipation
          :summary:
   * - :py:obj:`material <jaxmat.test_gsm.material>`
     - .. autodoc2-docstring:: jaxmat.test_gsm.material
          :summary:

API
~~~

.. py:function:: test_elastoplasticity(material, Nbatch=1)
   :canonical: jaxmat.test_gsm.test_elastoplasticity

   .. autodoc2-docstring:: jaxmat.test_gsm.test_elastoplasticity

.. py:data:: elastic_model
   :canonical: jaxmat.test_gsm.elastic_model
   :value: 'LinearElasticIsotropic(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.elastic_model

.. py:data:: viscous_model
   :canonical: jaxmat.test_gsm.viscous_model
   :value: 'LinearElasticIsotropic(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.viscous_model

.. py:data:: Nbatch
   :canonical: jaxmat.test_gsm.Nbatch
   :value: 'int(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.Nbatch

.. py:data:: free_energy
   :canonical: jaxmat.test_gsm.free_energy
   :value: 'FreeEnergy(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.free_energy

.. py:data:: dissipation
   :canonical: jaxmat.test_gsm.dissipation
   :value: 'DissipationPotential(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.dissipation

.. py:data:: material
   :canonical: jaxmat.test_gsm.material
   :value: 'GeneralizedStandardMaterial(...)'

   .. autodoc2-docstring:: jaxmat.test_gsm.material
