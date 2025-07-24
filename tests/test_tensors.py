import jax
import jax.numpy as jnp
from jaxmat.tensors.linear_algebra import eig33_HarariAlbocher, eig33
import numpy as np
import pytest


def random_unit_quaternions(key, batch_size):
    quat = jax.random.normal(key, (batch_size, 4))
    return quat / jnp.linalg.norm(quat, axis=-1, keepdims=True)


def quat_to_rotmat(q):
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.stack(
        [
            jnp.stack(
                [a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
                axis=-1,
            ),
            jnp.stack(
                [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2],
                axis=-1,
            ),
        ],
        axis=-2,
    )


def build_matrix_from_diag_and_quat(diag, quat):
    R = quat_to_rotmat(quat)
    D = jnp.diag(diag)
    A = R.T @ D @ R
    return A


def batch_build_A(diag, quats):
    return jax.jit(jax.vmap(build_matrix_from_diag_and_quat, in_axes=(None, 0)))(
        diag, quats
    )


def batch_eigvals(A_batch):
    return jax.vmap(eig33_HarariAlbocher)(A_batch)


diags_rand = jnp.array(np.random.rand(10, 3))
diags_two = jnp.array(
    [[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)]
)
diags_two = jnp.array(
    [[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)]
)
diags_three = jnp.array([[1, 1, 1]])  # , [0, 0, 0]])
diags = np.vstack((diags_rand, diags_two, diags_three))


@pytest.mark.parametrize("diag", diags)
def test_eigenvalue(diag):
    key = jax.random.PRNGKey(0)

    batch_size = int(10)

    quats = random_unit_quaternions(key, batch_size)

    # Build matrices A = Rᵀ D R
    A_batch = batch_build_A(diag, quats)
    for A in A_batch:
        eigvals, dyads = eig33_HarariAlbocher(A)
        A_reconstructed = sum([lamb * v for (lamb, v) in zip(eigvals, dyads)])
        assert np.allclose(jnp.sort(diag), eigvals)
        assert np.allclose(A, A_reconstructed)

    # test_batching
    eigvals_batch, dyads_batch = batch_eigvals(A_batch)

