:py:mod:`jaxmat.materials.viscoplasticity`
==========================================

.. py:module:: jaxmat.materials.viscoplasticity

.. autodoc2-docstring:: jaxmat.materials.viscoplasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AFInternalState <jaxmat.materials.viscoplasticity.AFInternalState>`
     - .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AFInternalState
          :summary:
   * - :py:obj:`AmrstrongFrederickViscoplasticity <jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity>`
     - .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity
          :summary:
   * - :py:obj:`GenericInternalState <jaxmat.materials.viscoplasticity.GenericInternalState>`
     - .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState
          :summary:
   * - :py:obj:`GenericViscoplasticity <jaxmat.materials.viscoplasticity.GenericViscoplasticity>`
     - .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity
          :summary:

API
~~~

.. py:class:: AFInternalState
   :canonical: jaxmat.materials.viscoplasticity.AFInternalState

   Bases: :py:obj:`jaxmat.state.SmallStrainState`

   .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AFInternalState

   .. py:attribute:: p
      :canonical: jaxmat.materials.viscoplasticity.AFInternalState.p
      :type: float
      :value: 'default_value(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AFInternalState.p

   .. py:attribute:: epsp
      :canonical: jaxmat.materials.viscoplasticity.AFInternalState.epsp
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AFInternalState.epsp

   .. py:attribute:: a
      :canonical: jaxmat.materials.viscoplasticity.AFInternalState.a
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AFInternalState.a

.. py:class:: AmrstrongFrederickViscoplasticity
   :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.elastic_model

   .. py:attribute:: yield_stress
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.yield_stress
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.yield_stress

   .. py:attribute:: viscous_flow
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.viscous_flow
      :type: jaxmat.materials.viscoplastic_flows.NortonFlow
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.viscous_flow

   .. py:attribute:: kinematic_hardening
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.kinematic_hardening
      :type: jaxmat.materials.viscoplastic_flows.ArmstrongFrederickHardening
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.kinematic_hardening

   .. py:attribute:: plastic_surface
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.plastic_surface
      :value: 'vonMises(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.plastic_surface

   .. py:attribute:: internal
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.internal
      :value: 'AFInternalState(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.internal

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.AmrstrongFrederickViscoplasticity.constitutive_update

.. py:class:: GenericInternalState
   :canonical: jaxmat.materials.viscoplasticity.GenericInternalState

   Bases: :py:obj:`jaxmat.state.SmallStrainState`

   .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState

   .. py:attribute:: p
      :canonical: jaxmat.materials.viscoplasticity.GenericInternalState.p
      :type: float
      :value: 'default_value(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState.p

   .. py:attribute:: epsp
      :canonical: jaxmat.materials.viscoplasticity.GenericInternalState.epsp
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState.epsp

   .. py:attribute:: nX
      :canonical: jaxmat.materials.viscoplasticity.GenericInternalState.nX
      :type: int
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState.nX

   .. py:attribute:: X
      :canonical: jaxmat.materials.viscoplasticity.GenericInternalState.X
      :type: jaxmat.tensors.SymmetricTensor2
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState.X

   .. py:method:: __post_init__()
      :canonical: jaxmat.materials.viscoplasticity.GenericInternalState.__post_init__

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericInternalState.__post_init__

.. py:class:: GenericViscoplasticity
   :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.elastic_model

   .. py:attribute:: plastic_surface
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.plastic_surface
      :type: jaxmat.materials.plastic_surfaces.AbstractPlasticSurface
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.plastic_surface

   .. py:attribute:: yield_stress
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.yield_stress
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.yield_stress

   .. py:attribute:: viscous_flow
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.viscous_flow
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.viscous_flow

   .. py:attribute:: kinematic_hardening
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.kinematic_hardening
      :type: jaxmat.materials.viscoplastic_flows.AbstractKinematicHardening
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.kinematic_hardening

   .. py:attribute:: internal
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.internal
      :type: jaxmat.state.AbstractState
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.internal

   .. py:method:: __post_init__()
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.__post_init__

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.__post_init__

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.viscoplasticity.GenericViscoplasticity.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.viscoplasticity.GenericViscoplasticity.constitutive_update
