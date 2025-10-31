import jax
import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2

import matplotlib.pyplot as plt


elasticity = jm.LinearElasticOrthotropic(EL=12.0e3, ET=0.8e3, EN=1.0e3, 
                                         nuLT = 0.43, nuLN = 0.47, nuTN = 0.292,
                                         muLT = 0.7e3, muLN = 0.9e3, muTN = 0.2e3)
material = jm.ElasticBehavior(elasticity=elasticity)
mat_state = material.init_state()

F = jnp.eye(3)

lamb = jnp.linspace(1, 1.1, 10)
N = len(lamb)

F = jnp.broadcast_to(F, (N, 3, 3))
F = F.at[:, 0, 0].set(lamb)
F = F.at[:, 1, 1].set(lamb)
F = F.at[:, 2, 2].set(lamb)

u_grad = F - jnp.eye(3)
eps_array = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))

eps = SymmetricTensor2(tensor=eps_array)

def compute_stress(eps_single):
    sig, _ = material.constitutive_update(eps_single, mat_state, dt=0.0)
    return sig

sig = jax.vmap(compute_stress)(eps)

C_tensor = elasticity.C

# L<->T permutation
angle = jnp.pi / 2 
axis = jnp.array([0., 0., 1.])
C_rotated_tensor = C_tensor.rotate_tensor(C_tensor.tensor, angle, axis)
from jaxmat.tensors import SymmetricTensor4
C_rotated = SymmetricTensor4(tensor=C_rotated_tensor)

elasticity_rotated = jm.LinearElastic(C=C_rotated)
material_rotated = jm.ElasticBehavior(elasticity=elasticity_rotated)


def compute_stress_rotated(eps_single):
    sig, _ = material_rotated.constitutive_update(eps_single, mat_state, dt=0.0)
    return sig

sig_rotated = jax.vmap(compute_stress_rotated)(eps)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Original material
ax1 = axes[0]
ax1.plot(lamb, sig[:, 0, 0], 'o-', label="σ₁₁ (L)", linewidth=2, markersize=6)
ax1.plot(lamb, sig[:, 1, 1], 's-', label="σ₂₂ (T)", linewidth=2, markersize=6)
ax1.plot(lamb, sig[:, 2, 2], '^-', label="σ₃₃ (N)", linewidth=2, markersize=6)
ax1.set_xlabel('λ', fontsize=12)
ax1.set_ylabel('Cauchy Stress (MPa)', fontsize=12)
ax1.set_title('Material (L-T-N)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Graphique 2: Matériau tourné de 90° autour de z
ax2 = axes[1]
ax2.plot(lamb, sig_rotated[:, 0, 0], 'o-', label="σ₁₁ (T→L)", linewidth=2, markersize=6)
ax2.plot(lamb, sig_rotated[:, 1, 1], 's-', label="σ₂₂ (L→T)", linewidth=2, markersize=6)
ax2.plot(lamb, sig_rotated[:, 2, 2], '^-', label="σ₃₃ (N)", linewidth=2, markersize=6)
ax2.set_xlabel('λ', fontsize=12)
ax2.set_ylabel('Cauchy stress(MPa)', fontsize=12)
ax2.set_title('Rotated material (L↔T permutation)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()