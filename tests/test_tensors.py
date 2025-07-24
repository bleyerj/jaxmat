import jax
import jax.numpy as jnp
from jaxmat.tensors.linear_algebra import (
    eig33,
    isotropic_function,
    stretch_tensor,
    polar,
    sqrtm,
    inv_sqrtm,
)
import numpy as np
import scipy.linalg as sl
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
    return jax.vmap(eig33)(A_batch)


diags_rand = jnp.array(np.random.rand(10, 3))
diags_two = jnp.array(
    [[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)]
)
diags_two = jnp.array(
    [[1, -0.5 + eps / 2, -0.5 - eps / 2] for eps in np.logspace(-3, -15, num=10)]
)
diags_three = jnp.array([[1, 1, 1]])  # , [0, 0, 0]])
diags = np.vstack((diags_rand, diags_two, diags_three))


@pytest.fixture(params=diags)
def diagonal(request):
    return request.param


@pytest.fixture
def quaternions():
    key = jax.random.PRNGKey(0)
    batch_size = int(10)
    return random_unit_quaternions(key, batch_size)


def test_eigenvalue(diagonal, quaternions):
    A_batch = batch_build_A(diagonal, quaternions)
    for A in A_batch:
        eigvals, dyads = eig33(A)
        A_reconstructed = sum([lamb * v for (lamb, v) in zip(eigvals, dyads)])
        assert np.allclose(jnp.sort(diagonal), eigvals)
        assert np.allclose(A, A_reconstructed)

    # test_batching
    eigvals_batch, dyads_batch = batch_eigvals(A_batch)


@pytest.mark.parametrize(
    ("matrix_fun", "scalar_fun"),
    [(sl.expm, jnp.exp), (sl.logm, jnp.log)],
)
def test_isotropic_function(matrix_fun, scalar_fun, diagonal, quaternions):
    A_batch = batch_build_A(jnp.abs(diagonal), quaternions)
    for A in A_batch:
        fA = matrix_fun(A)
        fA_ = isotropic_function(scalar_fun, A)
        assert np.allclose(fA, fA_)


def test_sqrtm(diagonal, quaternions):
    A_batch = batch_build_A(jnp.abs(diagonal), quaternions)
    for A in A_batch:
        fA = sqrtm(A)
        fA_ = isotropic_function(jnp.sqrt, A)
        assert np.allclose(fA, fA_)
        fA = inv_sqrtm(A)
        fA_ = isotropic_function(lambda x: 1 / jnp.sqrt(x), A)
        assert np.allclose(fA, fA_)


def test_stretch_tensor():
    gamma = 0.75
    Id = jnp.eye(3)
    F = jnp.array([[1, 1 + gamma, 0], [0, 1, 0], [0, 0, 1]])
    R, U = polar(F)
    C = F.T @ F
    B = F @ F.T
    assert jnp.allclose(F, R @ U)
    assert jnp.allclose(C, U @ U)
    assert jnp.allclose(Id, R.T @ R)
    V, R_ = polar(F, type="VR")
    assert jnp.allclose(R, R_)
    assert jnp.allclose(B, V @ V)
