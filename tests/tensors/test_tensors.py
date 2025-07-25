import pytest
import equinox as eqx
import jax.numpy as jnp
from jaxmat.tensors import SymmetricTensor2, Tensor2, polar, stretch_tensor
from jaxmat.tensors.linear_algebra import expm


def _tensor2_init(tensor_type, T_, T_vect_):
    T = tensor_type(tensor=T_)
    assert jnp.allclose(T, T_)
    assert jnp.allclose(T.array, T_vect_)
    T2 = tensor_type(array=T_vect_)
    assert jnp.allclose(T2, T_)
    assert jnp.allclose(T.T, T_.T)
    assert jnp.allclose(
        (T + T).array,
        2 * T_vect_,
    )
    assert jnp.allclose(
        (3 * T - T).array,
        2 * T_vect_,
    )
    assert jnp.allclose(
        (-T).array,
        -T_vect_,
    )
    assert jnp.allclose(
        T @ T,
        T_ @ T_,
    )


def test_tensor2_init():
    T_ = jnp.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=jnp.float64)
    T_vect_ = jnp.array([0, 4, 8, 1, 3, 2, 6, 5, 7], dtype=jnp.float64)
    _tensor2_init(Tensor2, T_, T_vect_)
    # check wrong size on initialization
    with pytest.raises(ValueError):
        SymmetricTensor2(array=T_vect_)
    # Warning: we don't check for symmetry upon initialization
    # But symmetry can be checked
    assert not SymmetricTensor2(tensor=T_).is_symmetric()


def test_sym_tensor2_init():
    S_ = jnp.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]], dtype=jnp.float64)
    S_vect_ = jnp.array(
        [0, 3, 5, jnp.sqrt(2) * 1, jnp.sqrt(2) * 2, jnp.sqrt(2) * 4], dtype=jnp.float64
    )
    # this passes
    Tensor2(tensor=S_)
    # this does not
    with pytest.raises(ValueError):
        Tensor2(array=S_vect_)

    S2_ = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=jnp.float64)
    S = SymmetricTensor2(tensor=S_)
    S2 = SymmetricTensor2(tensor=S2_)
    assert isinstance(S @ S2, Tensor2)
    assert not jnp.allclose(S @ S2, S2 @ S)
    assert isinstance((S @ S2).sym, SymmetricTensor2)
    assert jnp.allclose((S @ S2).sym, (S2 @ S).sym)


def test_stretch_tensor():
    gamma = 0.75
    Id = jnp.eye(3)
    F = Tensor2(
        tensor=jnp.array([[1, 1 + gamma, 0], [0, 1, 0], [0, 0, 1]], dtype=jnp.float64)
    )
    R, U = polar(F)
    C = (F.T @ F).sym
    B = (F @ F.T).sym
    # print(C, B)
    assert jnp.allclose(F, R @ U)
    assert jnp.allclose(C, U @ U)
    assert jnp.allclose(Id, R.T @ R)
    V, R_ = polar(F, mode="VR")
    assert jnp.allclose(R, R_)
    assert jnp.allclose(B, V @ V)
    U_ = stretch_tensor(F)
    assert jnp.allclose(U, U_)
