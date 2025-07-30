from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx
from . import linear_algebra


class Tensor(eqx.Module):
    dim: int
    rank: int
    _tensor: jax.Array

    def __init__(
        self, tensor: Optional[jax.Array] = None, array: Optional[jax.Array] = None
    ):

        if tensor is not None:
            if tensor.shape != self.shape:
                raise ValueError(f"Wrong shape {tensor.shape} <> {self.shape}")
            self._tensor = jnp.asarray(tensor)
        elif array is not None:
            if array.shape != self.array_shape:
                raise ValueError(f"Wrong shape {array.shape} <> {self.array_shape}")
            self._tensor = self._as_tensor(jnp.asarray(array))
        else:
            self._tensor = jnp.zeros(self.shape)

    @property
    def shape(self):
        return (self.dim,) * self.rank

    @property
    def tensor(self):
        return self._tensor

    @property
    def T(self):
        return self.__class__(tensor=jnp.transpose(self.tensor))

    @property
    def array(self):
        return self._as_array(self.tensor)

    @property
    def array_shape(self):
        return (self.dim**self.rank,)

    def __getitem__(self, idx):
        return self._tensor[idx]

    def __jax_array__(self):
        return self._tensor

    def __array__(self, dtype=None):
        return jnp.asarray(self._tensor, dtype=dtype)

    def __add__(self, other):
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor + other_array)

    def __sub__(self, other):
        cls = self._weaken_with(other)
        other_array = jnp.asarray(other).reshape(self.tensor.shape)
        return cls(tensor=self.tensor - other_array)

    def __mul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) * self.tensor)

    def __truediv__(self, other):
        return self.__class__(tensor=self.tensor / jnp.asarray(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__class__(tensor=jnp.asarray(self) @ jnp.asarray(other))

    def __rmatmul__(self, other):
        return self.__class__(tensor=jnp.asarray(other) @ self.tensor)

    def __neg__(self):
        return self.__class__(tensor=-self.tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}=\n{self.tensor}"

    def _as_array(self, tensor):
        return tensor.ravel()

    def _as_tensor(self, array):
        return array.reshape(self.shape)

    def _weaken_with(self, other):
        return self.__class__


class Tensor2(Tensor):
    dim = 3
    rank = 2

    @classmethod
    def identity(cls):
        return cls(tensor=jnp.eye(cls.dim))

    def _as_array(self, tensor):
        d = self.dim
        if tensor.ndim == 2:
            vec = [tensor[i, i] for i in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    vec.append(tensor[i, j])
                    vec.append(tensor[j, i])
            return jnp.array(vec)
        elif tensor.ndim == 3:
            vec = [tensor[:, i, i] for i in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    vec.append(tensor[:, i, j])
                    vec.append(tensor[:, j, i])
            return jnp.array(vec).T

    def _as_tensor(self, array):
        d = self.dim
        tensor = jnp.zeros((d, d))
        # Diagonal terms
        for i in range(d):
            tensor = tensor.at[i, i].set(array[i])

        # Off-diagonal terms
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                tensor = tensor.at[i, j].set(array[offset])
                tensor = tensor.at[j, i].set(array[offset + 1])
                offset += 2

        return tensor

    @property
    def sym(self):
        return SymmetricTensor2(tensor=0.5 * (self.tensor + self.tensor.T))

    @property
    def inv(self):
        return self.__class__(tensor=linear_algebra.inv33(self.tensor))

    @property
    def eigenvalues(self):
        eivenvalues, eigendyads = linear_algebra.eig33(self.tensor)
        return eivenvalues, jnp.asarray([SymmetricTensor2(N) for N in eigendyads])


class SymmetricTensor2(Tensor2):

    @property
    def array_shape(self):
        return (self.dim * (self.dim + 1) // 2,)

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    def _as_array(self, tensor):
        d = self.dim

        if tensor.ndim == 2:
            vec = [tensor[i, i] for i in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    vec.append(jnp.sqrt(2) * tensor[i, j])
            return jnp.array(vec)
        elif tensor.ndim == 3:
            vec = [tensor[:, i, i] for i in range(d)]
            for i in range(d):
                for j in range(i + 1, d):
                    vec.append(jnp.sqrt(2) * tensor[:, i, j])
            return jnp.array(vec).T

    def _as_tensor(self, array):
        d = self.dim
        tensor = jnp.zeros((d, d))

        # Diagonal entries
        for i in range(d):
            tensor = tensor.at[i, i].set(array[i])

        # Off-diagonal entries (upper triangle) scaled by 1/sqrt(2)
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                val = array[offset] / jnp.sqrt(2)
                tensor = tensor.at[i, j].set(val)
                tensor = tensor.at[j, i].set(val)  # symmetry
                offset += 1

        return tensor

    def __matmul__(self, other):
        # Multiplication of symmetric tensors cannot be ensured to remain symmetric
        return Tensor2(tensor=self.tensor @ jnp.asarray(other))

    def _weaken_with(self, other):
        if isinstance(other, self.__class__):
            return self.__class__
        return Tensor2


def symmetric_kelvin_mandel_index_map(d):
    """
    Returns:
        - km_to_ij: list mapping KM index → (i,j)
        - ij_to_km: dict mapping (i,j) → KM index
    """
    km_to_ij = []
    ij_to_km = {}
    idx = 0
    sqrt2 = 2**0.5
    for i in range(d):
        ij_to_km[(i, i)] = (idx, 1.0)
        km_to_ij.append(((i, i), 1.0))
        idx += 1
    for i in range(d):
        for j in range(i + 1, d):
            km_to_ij.append(((i, j), sqrt2))
            ij_to_km[(i, j)] = (idx, sqrt2)
            ij_to_km[(j, i)] = (idx, sqrt2)  # symmetry
            idx += 1
    return km_to_ij, ij_to_km


class SymmetricTensor4(Tensor):
    dim = 3
    rank = 4

    @classmethod
    def identity(cls):
        d = cls.dim
        n = d * (d + 1) // 2
        return cls(array=jnp.eye(n))

    @classmethod
    def J(cls):
        I2 = SymmetricTensor2.identity()
        J = jnp.einsum("ij,kl->ijkl", I2, I2) / cls.dim
        return cls(tensor=J)

    @classmethod
    def K(cls):
        return cls.identity() - cls.J()

    @property
    def array_shape(self):
        vdim = self.dim * (self.dim + 1) // 2
        return (vdim, vdim)

    def is_symmetric(self):
        return jnp.allclose(self, self.T)

    def _as_array(self, tensor: jax.Array) -> jax.Array:
        d = self.dim
        n = self.array_shape[0]
        km_to_ij, _ = symmetric_kelvin_mandel_index_map(d)
        array = jnp.zeros((n, n))

        for a, (ij, Na) in enumerate(km_to_ij):
            for b, (kl, Nb) in enumerate(km_to_ij):
                array = array.at[a, b].set(Na * Nb * tensor[*ij, *kl])
        return array

    def _as_tensor(self, array: jax.Array) -> jax.Array:
        """
        Converts a KM matrix (n,n) back to full symmetric 4th-order tensor (d,d,d,d)
        """
        d = self.dim
        km_to_ij, _ = symmetric_kelvin_mandel_index_map(d)
        tensor = jnp.zeros((d, d, d, d))

        for a, (ij, Na) in enumerate(km_to_ij):
            for b, (kl, Nb) in enumerate(km_to_ij):
                val = array[a, b] / (Na * Nb)

                # Assign to all symmetric permutations
                for ii, jj in {ij, reversed(ij)}:
                    for kk, ll in {kl, reversed(kl)}:
                        tensor = tensor.at[ii, jj, kk, ll].set(val)
                        tensor = tensor.at[kk, ll, ii, jj].set(val)
        return tensor

    def __matmul__(self, other):
        return other.__class__(
            tensor=jnp.tensordot(jnp.asarray(self), jnp.asarray(other).T)
        )

    @property
    def inv(self):
        return self.__class__(array=jnp.linalg.inv(self.array))


def _eval_basis(coeffs, basis):
    return sum([c * b for (c, b) in zip(coeffs, basis)])


class IsotropicTensor4(SymmetricTensor4):
    kappa: float
    mu: float

    def __init__(self, kappa, mu):
        self.kappa = kappa
        self.mu = mu
        super().__init__(self.eval())

    @property
    def basis(self):
        J = SymmetricTensor4.J()
        K = SymmetricTensor4.K()
        return [J, K]

    @property
    def coeffs(self):
        return jnp.asarray([3 * self.kappa, 2 * self.mu])

    def eval(self):
        return _eval_basis(self.coeffs, self.basis)

    @property
    def inv(self):
        return IsotropicTensor4(1 / 9 / self.kappa, 1 / 4 / self.mu)
