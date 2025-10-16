:py:mod:`jaxmat.materials.elasticity`
=====================================

.. py:module:: jaxmat.materials.elasticity

.. autodoc2-docstring:: jaxmat.materials.elasticity
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LinearElasticIsotropic <jaxmat.materials.elasticity.LinearElasticIsotropic>`
     -
   * - :py:obj:`ElasticBehavior <jaxmat.materials.elasticity.ElasticBehavior>`
     -

API
~~~

.. py:class:: LinearElasticIsotropic
   :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic

   Bases: :py:obj:`equinox.Module`

   .. py:attribute:: E
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.E
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.E

   .. py:attribute:: nu
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.nu
      :type: float
      :value: 'field(...)'

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.nu

   .. py:attribute:: internal
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.internal
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.internal

   .. py:property:: kappa
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.kappa

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.kappa

   .. py:property:: mu
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.mu

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.mu

   .. py:property:: C
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.C

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.C

   .. py:property:: S
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.S

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.S

   .. py:method:: strain_energy(eps)
      :canonical: jaxmat.materials.elasticity.LinearElasticIsotropic.strain_energy

      .. autodoc2-docstring:: jaxmat.materials.elasticity.LinearElasticIsotropic.strain_energy

.. py:class:: ElasticBehavior
   :canonical: jaxmat.materials.elasticity.ElasticBehavior

   Bases: :py:obj:`jaxmat.materials.behavior.SmallStrainBehavior`

   .. py:attribute:: elasticity
      :canonical: jaxmat.materials.elasticity.ElasticBehavior.elasticity
      :type: equinox.Module
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elasticity.ElasticBehavior.elasticity

   .. py:attribute:: internal
      :canonical: jaxmat.materials.elasticity.ElasticBehavior.internal
      :value: None

      .. autodoc2-docstring:: jaxmat.materials.elasticity.ElasticBehavior.internal

   .. py:method:: constitutive_update(eps, state, dt)
      :canonical: jaxmat.materials.elasticity.ElasticBehavior.constitutive_update

      .. autodoc2-docstring:: jaxmat.materials.elasticity.ElasticBehavior.constitutive_update
