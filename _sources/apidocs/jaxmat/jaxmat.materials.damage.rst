:py:mod:`jaxmat.materials.damage`
=================================

.. py:module:: jaxmat.materials.damage

.. autodoc2-docstring:: jaxmat.materials.damage
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`InternalState <jaxmat.materials.damage.InternalState>`
     -
   * - :py:obj:`IsotropicDegradation <jaxmat.materials.damage.IsotropicDegradation>`
     -
   * - :py:obj:`DamageThreshold <jaxmat.materials.damage.DamageThreshold>`
     -
   * - :py:obj:`Damage <jaxmat.materials.damage.Damage>`
     -

API
~~~

.. py:class:: InternalState
   :canonical: jaxmat.materials.damage.InternalState

   Bases: :py:obj:`jaxmat.state.AbstractState`

   .. py:attribute:: d
      :canonical: jaxmat.materials.damage.InternalState.d
      :type: float
      :value: 'default_value(...)'

      .. autodoc2-docstring:: jaxmat.materials.damage.InternalState.d

.. py:class:: IsotropicDegradation
   :canonical: jaxmat.materials.damage.IsotropicDegradation

   Bases: :py:obj:`equinox.Module`

   .. py:method:: __call__(d)
      :canonical: jaxmat.materials.damage.IsotropicDegradation.__call__

      .. autodoc2-docstring:: jaxmat.materials.damage.IsotropicDegradation.__call__

.. py:class:: DamageThreshold
   :canonical: jaxmat.materials.damage.DamageThreshold

   Bases: :py:obj:`equinox.Module`

   .. py:attribute:: Y0
      :canonical: jaxmat.materials.damage.DamageThreshold.Y0
      :type: float
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.damage.DamageThreshold.Y0

   .. py:attribute:: alpha
      :canonical: jaxmat.materials.damage.DamageThreshold.alpha
      :type: float
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.damage.DamageThreshold.alpha

   .. py:method:: __call__(d)
      :canonical: jaxmat.materials.damage.DamageThreshold.__call__

      .. autodoc2-docstring:: jaxmat.materials.damage.DamageThreshold.__call__

.. py:class:: Damage
   :canonical: jaxmat.materials.damage.Damage

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. py:attribute:: elastic_model
      :canonical: jaxmat.materials.damage.Damage.elastic_model
      :type: jaxmat.materials.elasticity.LinearElasticIsotropic
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.damage.Damage.elastic_model

   .. py:attribute:: degradation
      :canonical: jaxmat.materials.damage.Damage.degradation
      :type: jaxmat.materials.damage.IsotropicDegradation
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.damage.Damage.degradation

   .. py:attribute:: damage_threshold
      :canonical: jaxmat.materials.damage.Damage.damage_threshold
      :type: jaxmat.materials.damage.DamageThreshold
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.damage.Damage.damage_threshold

   .. py:attribute:: internal
      :canonical: jaxmat.materials.damage.Damage.internal
      :type: jaxmat.state.AbstractState
      :value: 'InternalState(...)'

      .. autodoc2-docstring:: jaxmat.materials.damage.Damage.internal

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.damage.Damage.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.damage.Damage.constitutive_update
