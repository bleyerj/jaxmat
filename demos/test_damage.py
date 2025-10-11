# %%
import jax

jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import equinox as eqx
import jaxmat.materials as jm
import matplotlib.pyplot as plt
from jaxmat.materials.damage import Damage, DamageThreshold, IsotropicDegradation

elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
degradation = IsotropicDegradation()
threshold = DamageThreshold(Y0=2e-3, alpha=1.0)

material = Damage(
    elastic_model=elasticity, degradation=degradation, damage_threshold=threshold
)
print(material.elastic_model.__dict__)
print(material)

mu = elasticity.mu
print(f"\nShear modulus = {1e-3*mu} GPa")

# %%
state = material.init_state()
print(state.__dict__)
internal_state_variables = material.internal
print(internal_state_variables.__dict__)

# %%
from jaxmat.tensors import SymmetricTensor2

dt = 0.0

# %%
gamma_load = jnp.linspace(0, 2e-4, 50)
gamma_list = jnp.concatenate((gamma_load, gamma_load[::-1][1:], 2 * gamma_load))
# gamma_list = gamma_load
state = material.init_state()
tau = jnp.zeros_like(gamma_list)
for i, gamma in enumerate(gamma_list):
    new_eps = jnp.array([[0, gamma / 2, 0], [gamma / 2, 0, 0], [0, 0, 0]])
    new_eps = SymmetricTensor2(tensor=new_eps)
    dt = 0.0
    new_stress, new_state = material.constitutive_update(new_eps, state, dt)
    state = new_state
    print(new_stress)
    tau = tau.at[i].set(new_stress[0, 1])

# %%
plt.plot(gamma_list, tau, "-xk")
plt.xlabel(r"Shear distorsion $\gamma$")
plt.ylabel(r"Shear stress $\tau$ [MPa]")
plt.show()
