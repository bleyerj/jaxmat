import jax
import jax.numpy as jnp
import jaxmat.materials as jm
from jaxmat.tensors import Tensor2, SymmetricTensor2

F = jnp.eye(3)
material = jm.CompressibleOgden(mu=4.0, alpha=jnp.array([2.0, -2.0]), kappa=1e3)

lamb = jnp.linspace(1, 2.5, 10)
N = len(lamb)


F = jnp.broadcast_to(F, (N, 3, 3))
F = F.at[:, 0, 0].set(lamb)
F = F.at[:, 1, 1].set(lamb)
F = F.at[:, 2, 2].set(1 / lamb**2)

F = Tensor2(tensor=F)

P = jax.vmap(material.PK1)(F)
sig = jax.vmap(material.Cauchy)(F)

import matplotlib.pyplot as plt

plt.plot(F[:, 0, 0], P[:, 0, 0], label="PK1")
plt.plot(F[:, 0, 0], sig[:, 0, 0], label="Cauchy")
plt.show()
