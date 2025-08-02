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

# +
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import equinox as eqx
import jaxmat.materials as jm
import matplotlib.pyplot as plt



class VoceHardening(eqx.Module):
    sig0: float
    sigu: float
    b: float
    def __call__(self, p):
        return self.sig0 + (self.sigu-self.sig0)*(1-jnp.exp(-self.b*p))


elasticity = jm.LinearElasticIsotropic(E=200e3, nu=0.25)
hardening = VoceHardening(sig0=350., sigu=500.0, b=1e3)

material = jm.vonMisesIsotropicHardening(elastic_model=elasticity, yield_stress=hardening)
print(material.elastic_model.__dict__)
print(material.yield_stress.__dict__)
print(material)

mu = elasticity.mu
print(f"\nShear modulus = {1e-3*mu} GPa")

# +
hardening_modulus = jax.grad(hardening)

p = jnp.linspace(0, 5e-3, 100)
H = jax.vmap(hardening_modulus)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(p, hardening(p), '-C0')
plt.gca().set_ylim(bottom=0)
plt.xlabel("Equivalent plastic strain $p$")
plt.ylabel("Yield stress $R(p)$ [MPa]");
plt.subplot(1, 2, 2)
plt.plot(p, 1e-3*H(p), '-C3')
plt.gca().set_ylim(bottom=0)
plt.xlabel("Equivalent plastic strain $p$")
plt.ylabel("Hardening modulus $R'(p)$ [GPa]");
# -

state = material.init_state()
print(state.__dict__)
internal_state_variables = material.internal
print(internal_state_variables.__dict__)

from jaxmat.tensors import SymmetricTensor2
gamma = 1e-3
new_eps = jnp.array([[0, gamma/2, 0], 
                     [gamma/2, 0, 0], 
                     [0, 0, 0]])
new_eps = SymmetricTensor2(tensor=new_eps)
dt=0.0
new_stress, new_state = material.constitutive_update(new_eps, state, dt)
print(new_stress)
print(new_state.__dict__)

gamma_list = jnp.linspace(0, 1e-2, 100)
state = material.init_state()
tau = jnp.zeros_like(gamma_list)
for i, gamma in enumerate(gamma_list):
    new_eps = jnp.array([[0, gamma/2, 0], 
                     [gamma/2, 0, 0], 
                     [0, 0, 0]])
    new_eps = SymmetricTensor2(tensor=new_eps)
    dt=0.0
    new_stress, new_state = material.constitutive_update(new_eps, state, dt)
    state = new_state
    tau = tau.at[i].set(new_stress[0, 1])

plt.plot(gamma_list, tau, "-k")
plt.xlabel(r"Shear distorsion $\gamma$")
plt.ylabel(r"Shear stress $\tau$ [MPa]");

tangent_operator = jax.jacfwd(material.constitutive_update, argnums=0, has_aux=True)

# Let us now do the exact same calculation as before except that we call `tangent_operator` which now returns a tuple `(Ctang, new_state)` containing the tangent operator $\mathbb{C}_\text{tang}$ and the new material state. We have formally replaced the first output of `constitutive_update` with its Jacobian with respect to the applied strain. As a result, the stress is not directly returned as before. We can however retrieve it from the new state as `new_state.stress`.

gamma_list = jnp.linspace(0, 1e-2, 100)
state = material.init_state()
tau = jnp.zeros_like(gamma_list)
p = jnp.zeros_like(gamma_list)
mu_tang = jnp.zeros_like(gamma_list)
for i, gamma in enumerate(gamma_list):
    new_eps = jnp.array([[0, gamma/2, 0], 
                     [gamma/2, 0, 0], 
                     [0, 0, 0]])
    new_eps = SymmetricTensor2(tensor=new_eps)
    dt=0.0
    Ctang, new_state = tangent_operator(new_eps, state, dt)
    state = new_state
    new_stress = state.stress
    tau = tau.at[i].set(new_stress[0, 1])
    p = p.at[i].set(state.internal.p)
    mu_tang = mu_tang.at[i].set(Ctang[0, 1, 0, 1])

# Note that the tangent operator is the derivative of a `SymmetricTensor2` with respect to a `SymmetricTensor2` which is formally equivalent to a 4th-rank tensor. Its component can thus be accessed as `Ctang[i, j, k, l]`. Collecting the value of the tangent shear modulus from `Ctang[0, 1, 0, 1]`, its evolution clearly shows a first constant phase in the elastic regime where $\mu_\text{tang}=\mu=80\text{ GPa}$. We see the sudden drop at the onset of plasticity which is due to the finite initial hardening slope $R'(0)$. We now that the material (elastoplastic) tangent operator is given by:
#
# $$\mathbb{C}^\text{ep} = 3\kappa\mathbb{J} + 2\mu\left(\mathbb{K}-\dfrac{3\mu}{3\mu+R'(p)}\boldsymbol{n}\otimes\boldsymbol{n}\right)$$
# where $\boldsymbol{n}$ is the unit normal vector in the direction of the plastic flow. For pure shear conditions, this gives the elastoplastic tangent shear modulus:
# $$
# \mu^\text{ep} = \mu\left(1-\dfrac{3\mu}{3\mu+R'(p)}\right) = \mu\dfrac{R'(p)}{3\mu+R'(p)}
# $$ 

plt.plot(gamma_list, mu_tang*1e-3, "-k", label="Consistent")
mu_ep = jnp.full_like(gamma_list, mu)
plastic_points = jnp.where(p>1e-8)
H_plast = H(p[plastic_points])
mu_ep = mu_ep.at[plastic_points].set(mu*(H_plast/(3*mu+H_plast)))
plt.plot(gamma_list, 1e-3*mu_ep, "xC3", label="Material")
plt.ylim(0, 90)
plt.xlabel(r"Shear distorsion $\gamma$")
plt.ylabel(r"Tangent shear modulus $\mu_\textrm{tang}$ [GPa]")
plt.legend();

# ## Computation for a batch of material points
#
# In this section, we will show how to adapt the previous setting to the evaluation of the constitutive law for a set of $N$ material points, which we will call a *batch* of size $N$. To do so, we will heavily rely on `jax`'s automatic vectorization functionality provided by the `jax.vmap` function. 
#
# As an illustration, let us consider here the case of perfect plasticity and perform a single evaluation of the constitutive update for points with imposed strains such that the elastic prediction will fall outside the yield surface. The result of the constitutive update will therefore produce points which are projected onto the yield surface. We consider purely deviatoric strains of the form:
#
# $$
# \boldsymbol{\varepsilon}=\text{diag}(e_I, e_{II}, -(e_I+e_{II}))
# $$
#
# The batch of points will consist of $N$ values such that $e_I = \epsilon\cos(\theta_k), e_{II}=\epsilon\sin(\theta_k)$ for $\theta_k \in [0;2\pi]$ and $k=1,\ldots,N$. Here the amplitude $\epsilon$ is fixed and chosen sufficiently large to fall outside the plastic yield surface.
#
# We first represent $\boldsymbol{\varepsilon}$ as a batched `SymmetricTensor2` of shape `(N,3,3)`. By convention, the batch dimension is always the first one.

# +
N = 40
theta = jnp.linspace(0, jnp.pi, N)

eps_ = 2e-3
eps = jnp.zeros((N, 3, 3))
eps = eps.at[:, 0, 0].set(eps_*jnp.cos(theta))
eps = eps.at[:, 1, 1].set(eps_*jnp.sin(theta))
eps = eps.at[:, 2, 2].set(-eps[:, 0, 0]-eps[:,1,1])
eps = SymmetricTensor2(tensor=eps)

# +
sig0 = 300.0
class YieldStress(eqx.Module):
    sig0: float
    H_: float = 1e-6
    def __call__(self, p):
        return  self.sig0*(1.0+self.H_*p)
    
new_material = jm.GeneralIsotropicHardening(elastic_model=jm.LinearElasticIsotropic(E=200e3, nu=0), yield_stress=YieldStress(sig0=sig0), plastic_surface=jm.Hosford())
state = new_material.init_state(Nbatch=N)
# -

batched_constitutive_update = jax.vmap(jm.GeneralIsotropicHardening.constitutive_update, in_axes=(None, 0, 0, None))


def scatter_pi_plane(stress, marker="o", **kwargs):
    from jaxmat.tensors import eigenvalues
    eigvals = jax.vmap(eigenvalues, in_axes=0)(stress)
    xx = jnp.concatenate([eigvals[:, i]-eigvals[:, (i+2)%3] for i in range(3)])
    xx = jnp.concatenate((xx, -xx))*jnp.sqrt(3)/2
    yy = jnp.concatenate([eigvals[:, (i+1)%3] for i in range(3)])
    yy = jnp.concatenate((yy, yy))*3/2
    plt.scatter(xx, yy, marker=marker, **kwargs)
    margin = 0.1
    lim = (1+margin)*sig0
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.gca().set_aspect("equal")


plt.figure(figsize=(6, 6))
for i, a in enumerate([2.0, 6.0, 10.0]):
    new_material = eqx.tree_at(lambda m: m.plastic_surface.a, new_material, jnp.asarray(a))
    stress, new_state = batched_constitutive_update(new_material, eps, state, 0.0)
    scatter_pi_plane(stress, "x", color=f"C{i}", linewidth=0.5, label=rf"$a={int(a)}$")
plt.legend();
