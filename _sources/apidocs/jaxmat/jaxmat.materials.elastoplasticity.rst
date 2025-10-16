:py:mod:`jaxmat.materials.elastoplasticity`
===========================================

.. py:module:: jaxmat.materials.elastoplasticity

.. autodoc2-docstring:: jaxmat.materials.elastoplasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InternalState <jaxmat.materials.elastoplasticity.InternalState>`
     - .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.InternalState
          :summary:
   * - :py:obj:`vonMisesIsotropicHardening <jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening>`
     -
   * - :py:obj:`GeneralIsotropicHardening <jaxmat.materials.elastoplasticity.GeneralIsotropicHardening>`
     -

API
~~~

.. py:class:: InternalState
   :canonical: jaxmat.materials.elastoplasticity.InternalState

   Bases: :py:obj:`jaxmat.state.AbstractState`

   .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.InternalState

   .. py:attribute:: p
      :canonical: jaxmat.materials.elastoplasticity.InternalState.p
      :type: jax.Array
      :value: 'default_value(...)'

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.InternalState.p

   .. py:attribute:: epsp
      :canonical: jaxmat.materials.elastoplasticity.InternalState.epsp
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.InternalState.epsp

.. py:class:: vonMisesIsotropicHardening
   :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.elastic_model

   .. py:attribute:: yield_stress
      :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.yield_stress
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.yield_stress

   .. py:attribute:: plastic_surface
      :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.plastic_surface
      :type: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
      :value: 'vonMises(...)'

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.plastic_surface

   .. py:attribute:: internal
      :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.internal
      :type: jaxmat.state.AbstractState
      :value: 'InternalState(...)'

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.internal

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.vonMisesIsotropicHardening.constitutive_update

.. py:class:: GeneralIsotropicHardening
   :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.elastic_model

   .. py:attribute:: yield_stress
      :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.yield_stress
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.yield_stress

   .. py:attribute:: plastic_surface
      :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.plastic_surface
      :type: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.plastic_surface

   .. py:attribute:: internal
      :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.internal
      :value: 'InternalState(...)'

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.internal

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.elastoplasticity.GeneralIsotropicHardening.constitutive_update
