import jax.numpy as jnp
import equinox as eqx
from jaxmat.state import (
    make_batched,
)
import jaxmat.materials as jm


class YieldStress(eqx.Module):
    sig0: float

    def __call__(self, p):
        return self.sig0


def test_small_strain_behavior():
    elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
    hardening = YieldStress(sig0=300.0)

    material = jm.vonMisesIsotropicHardening(
        elastic_model=elasticity, yield_stress=hardening
    )
    state = material.init_state()
    assert hasattr(state, "strain")
    assert hasattr(state, "strain")
    assert hasattr(state.internal, "p")
    assert hasattr(state.internal, "epsp")

    Nbatch = 10
    batched_state = material.init_state(Nbatch)
    assert jnp.array_equal(batched_state.strain, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.internal.epsp, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(batched_state.internal.p, jnp.zeros((Nbatch,)))

    material2 = jm.vonMisesIsotropicHardening(
        elastic_model=elasticity, yield_stress=hardening
    )
    batched_material = make_batched(material2, Nbatch)
    assert jnp.array_equal(
        batched_material.elastic_model.E, jnp.full((Nbatch,), elasticity.E)
    )
    state = batched_material.init_state()
    assert jnp.array_equal(state.strain, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.epsp, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(state.internal.p, jnp.zeros((Nbatch,)))


def test_finite_strain_behavior():
    elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
    hardening = YieldStress(sig0=300.0)

    material = jm.FeFpJ2Plasticity(elastic_model=elasticity, yield_stress=hardening)
    state = material.init_state()
    assert hasattr(state, "strain")
    assert hasattr(state, "strain")
    assert hasattr(state.internal, "p")
    assert hasattr(state.internal, "be_bar")

    Nbatch = 10
    batched_state = material.init_state(Nbatch)
    assert jnp.array_equal(
        batched_state.strain, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3))
    )
    assert jnp.array_equal(batched_state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(
        batched_state.internal.be_bar, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3))
    )
    assert jnp.array_equal(batched_state.internal.p, jnp.zeros((Nbatch,)))

    material2 = jm.FeFpJ2Plasticity(elastic_model=elasticity, yield_stress=hardening)
    batched_material = make_batched(material2, Nbatch)
    assert jnp.array_equal(
        batched_material.elastic_model.E, jnp.full((Nbatch,), elasticity.E)
    )
    state = batched_material.init_state()
    assert jnp.array_equal(state.strain, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3)))
    assert jnp.array_equal(state.stress, jnp.zeros((Nbatch, 3, 3)))
    assert jnp.array_equal(
        state.internal.be_bar, jnp.broadcast_to(jnp.eye(3), (Nbatch, 3, 3))
    )
    assert jnp.array_equal(state.internal.p, jnp.zeros((Nbatch,)))
