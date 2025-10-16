:py:mod:`jaxmat.materials.fe_fp_elastoplasticity`
=================================================

.. py:module:: jaxmat.materials.fe_fp_elastoplasticity

.. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InternalState <jaxmat.materials.fe_fp_elastoplasticity.InternalState>`
     -
   * - :py:obj:`FeFpJ2Plasticity <jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity>`
     - .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity
          :summary:

API
~~~

.. py:class:: InternalState
   :canonical: jaxmat.materials.fe_fp_elastoplasticity.InternalState

   Bases: :py:obj:`jaxmat.state.AbstractState`

   .. py:attribute:: p
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.InternalState.p
      :type: float
      :value: 'default_value(...)'

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.InternalState.p

   .. py:attribute:: be_bar
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.InternalState.be_bar
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'identity(...)'

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.InternalState.be_bar

.. py:class:: FeFpJ2Plasticity
   :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity

   Bases: :py:obj:`jaxmat.materials.behavior.FiniteStrainBehavior`

   .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.elastic_model

   .. py:attribute:: yield_stress
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.yield_stress
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.yield_stress

   .. py:attribute:: plastic_surface
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.plastic_surface
      :type: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
      :value: 'vonMises(...)'

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.plastic_surface

   .. py:attribute:: internal
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.internal
      :type: jaxmat.state.AbstractState
      :value: 'InternalState(...)'

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.internal

   .. py:attribute:: solver
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.solver
      :value: 'LevenbergMarquardt(...)'

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.solver

   .. py:method:: constitutive_update(F, state, dt)
      :canonical: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.fe_fp_elastoplasticity.FeFpJ2Plasticity.constitutive_update
