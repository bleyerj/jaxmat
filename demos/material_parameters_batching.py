# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: fenicsx-v0.9
#     language: python
#     name: python3
# ---

# ## Batching across material parameters
#
# In the previous section, we batched over applied strain and state variable, assuming constant material properties. We show how here how we can easily adapt the previous idea to a configuration where we batch over material parameters at the same time.
#
# We go back to our initial elastoplastic material with Voce hardening. We want to investigate this effect of uncertainty on the hardening parameters for a monotonous shear loading. In this example, we consider the response of 1 physical point only under the same applied loading. However, since we will sample for $N$ independent realizations of material properties, we will in fact compute the response of a batch of $N$ stochastic material points.

# +
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import equinox as eqx
import jaxmat.materials as jm
import matplotlib.pyplot as plt
from jaxmat.tensors import SymmetricTensor2
from jaxmat.state import make_batched

class VoceHardening(eqx.Module):
    sig0: float
    sigu: float
    b: float
    def __call__(self, p):
        return self.sig0 + (self.sigu-self.sig0)*(1-jnp.exp(-self.b*p))


elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
hardening = VoceHardening(sig0=350., sigu=500.0, b=1e3)

material = jm.vonMisesIsotropicHardening(elastic_model=elasticity, yield_stress=hardening)
# material = jm.ElasticBehavior(elasticity)

print("A single material instance:", material)

key = jax.random.PRNGKey(42)
N = 50
b_values = 10**(3*jax.random.lognormal(key, sigma=0.05, shape=(N,)))
sorting = jnp.argsort(b_values)

batched_material = make_batched(material, N)
batched_material = eqx.tree_at(lambda m: m.yield_stress.b, batched_material, b_values)
print(f"A batch of {N} material instances:", batched_material)
# state = batched_material.init_state()
# def eval_C(material):
#     jax.debug.print("material={}", material.elasticity.E)
#     return material.elasticity.C

# C = eqx.filter_vmap(eval_C, in_axes=0)(batched_material)
# print(C.tensor.shape)

# +
state = make_batched(material.init_state(), N) # FIXME: we should be able to do batched_material.init_state(), right now isv are batched once more

gamma_list = jnp.linspace(0, 1e-2, 100)
tau = jnp.zeros((N, len(gamma_list)))
for i, gamma in enumerate(gamma_list):
    new_eps = jnp.array([[0, gamma/2, 0], 
                     [gamma/2, 0, 0], 
                     [0, 0, 0]])
    new_eps = SymmetricTensor2(tensor=new_eps)
    dt=0.0
    new_stress, new_state = eqx.filter_vmap(jm.vonMisesIsotropicHardening.constitutive_update, in_axes=(0, None, 0, None))(batched_material, new_eps, state, 0.0)
    state = new_state
    tau = tau.at[:, i].set(new_stress[:, 0, 1])
# -

cmap = plt.get_cmap("bwr")
colors = cmap(jnp.linspace(0, 1, N))
for i, color in enumerate(colors):
    plt.plot(gamma_list, tau[sorting[i], :].T, linewidth=1.0, alpha=0.5, color=colors[i])
plt.xlabel(r"Shear distorsion $\gamma$")
plt.ylabel(r"Shear stress $\tau$ [MPa]");




