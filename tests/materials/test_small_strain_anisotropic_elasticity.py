import jax
import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import SymmetricTensor2
import matplotlib.pyplot as plt

# Define orthotropic elastic material
elasticity = jm.LinearElasticOrthotropic(EL=12.0e3, ET=0.8e3, EN=1.0e3, 
                                         nuLT=0.43, nuLN=0.47, nuTN=0.292,
                                         muLT=0.7e3, muLN=0.9e3, muTN=0.2e3)
material = jm.ElasticBehavior(elasticity=elasticity)
mat_state = material.init_state()

# Create deformation gradient tensor
F = jnp.eye(3)
lamb = jnp.linspace(1, 1.1, 10)
N = len(lamb)
F = jnp.broadcast_to(F, (N, 3, 3))
F = F.at[:, 0, 0].set(lamb)
F = F.at[:, 1, 1].set(lamb)
F = F.at[:, 2, 2].set(lamb)

# Compute strain tensor
u_grad = F - jnp.eye(3)
eps_array = 0.5 * (u_grad + jnp.swapaxes(u_grad, -1, -2))
eps = SymmetricTensor2(tensor=eps_array)

# Compute stress for original material
def compute_stress(eps_single):
    sig, _ = material.constitutive_update(eps_single, mat_state, dt=0.0)
    return sig

sig = jax.vmap(compute_stress)(eps)

# Rotate stiffness tensor: L<->T permutation (90° around z-axis)
C_tensor = elasticity.C
angle = jnp.pi / 2 
axis = jnp.array([0., 0., 1.])

from jaxmat.tensors import utils
R = utils.rotation_matrix_direct(angle, axis)
        
C_rotated_tensor = C_tensor.rotate_tensor(C_tensor.tensor, R)

from jaxmat.tensors import SymmetricTensor4
C_rotated = SymmetricTensor4(tensor=C_rotated_tensor)
elasticity_rotated = jm.LinearElastic(C=C_rotated)
material_rotated = jm.ElasticBehavior(elasticity=elasticity_rotated)

# Compute stress for rotated material
def compute_stress_rotated(eps_single):
    sig, _ = material_rotated.constitutive_update(eps_single, mat_state, dt=0.0)
    return sig

sig_rotated = jax.vmap(compute_stress_rotated)(eps)

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Graph 1: Original material
ax1 = axes[0]
ax1.plot(lamb, sig[:, 0, 0], 'o-', label="σ₁₁ (L)", linewidth=2, markersize=6)
ax1.plot(lamb, sig[:, 1, 1], 's-', label="σ₂₂ (T)", linewidth=2, markersize=6)
ax1.plot(lamb, sig[:, 2, 2], '^-', label="σ₃₃ (N)", linewidth=2, markersize=6)
ax1.set_xlabel('λ', fontsize=12)
ax1.set_ylabel('Cauchy Stress (MPa)', fontsize=12)
ax1.set_title('Original Material (L-T-N)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Graph 2: Material rotated 90° around z-axis
ax2 = axes[1]
ax2.plot(lamb, sig_rotated[:, 0, 0], 'o-', label="σ₁₁ (T→L)", linewidth=2, markersize=6)
ax2.plot(lamb, sig_rotated[:, 1, 1], 's-', label="σ₂₂ (L→T)", linewidth=2, markersize=6)
ax2.plot(lamb, sig_rotated[:, 2, 2], '^-', label="σ₃₃ (N)", linewidth=2, markersize=6)
ax2.set_xlabel('λ', fontsize=12)
ax2.set_ylabel('Cauchy Stress (MPa)', fontsize=12)
ax2.set_title('Rotated Material (L↔T Permutation)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Verify that stress components are correctly permuted
assert jnp.allclose(sig[:, 0, 0], sig_rotated[:, 1, 1], atol=1e-6)
assert jnp.allclose(sig[:, 1, 1], sig_rotated[:, 0, 0], atol=1e-6)