from typing import Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx


class Tensor(eqx.Module):
    dim: int
    rank: int
    shape: tuple
    _tensor: jax.Array

    def __init__(
        self, tensor: Optional[jax.Array] = None, array: Optional[jax.Array] = None
    ):

        if tensor is not None:
            if tensor.shape != self.shape:
                raise ValueError("Wrong shape {tensor.shape} <> {self.shape}")
            self._tensor = tensor
            self.array  # force call to do some sanity checks
        elif array is not None:
            if array.shape != self.array_shape:
                raise ValueError(f"Wrong shape {array.shape} <> {self.array_shape}")
            self._tensor = self._as_tensor(array)
        else:
            self._tensor = jnp.zeros(self.shape)

    @property
    def tensor(self):
        return self._tensor

    @property
    def T(self):
        return jnp.transpose(self.tensor)

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
        return np.asarray(self._tensor, dtype=dtype)

    def __add__(self, other):
        return self.__class__(tensor=self.tensor + other.tensor)

    def __sub__(self, other):
        return self.__class__(tensor=self.tensor - other.tensor)

    def __mul__(self, other):
        return self.__class__(tensor=other * self.tensor)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return self.__class__(tensor=self.tensor @ other.tensor)

    def __neg__(self):
        return self.__class__(tensor=-self.tensor)

    def __repr__(self):
        return f"{self.__class__.__name__}=\n{self.tensor}"

    def _as_array(self, tensor):
        return tensor.ravel()

    def _as_tensor(self, vector):
        return vector.reshape(self.shape)


class Tensor2(Tensor):
    dim = 3
    rank = 2
    shape = (dim, dim)

    def _as_array(self, tensor):
        d = self.dim
        vec = [tensor[i, i] for i in range(d)]
        for i in range(d):
            for j in range(i + 1, d):
                vec.append(tensor[i, j])
                vec.append(tensor[j, i])
        return jnp.array(vec)

    def _as_tensor(self, vector):
        d = self.dim
        tensor = jnp.zeros((d, d))
        # Diagonal terms
        for i in range(d):
            tensor = tensor.at[i, i].set(vector[i])

        # Off-diagonal terms
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                tensor = tensor.at[i, j].set(vector[offset])
                tensor = tensor.at[j, i].set(vector[offset + 1])
                offset += 2

        return tensor

    @property
    def sym(self):
        return SymmetricTensor2(tensor=0.5 * (self.tensor + self.tensor.T))


class SymmetricTensor2(Tensor2):

    @property
    def array_shape(self):
        return (self.dim * (self.dim + 1) // 2,)

    def _as_array(self, tensor):
        if not jnp.allclose(tensor, tensor.T):
            raise ValueError("Tensor is not symmetric.")
        d = self.dim
        vec = [tensor[i, i] for i in range(d)]
        for i in range(d):
            for j in range(i + 1, d):
                vec.append(jnp.sqrt(2) * tensor[i, j])
        return jnp.array(vec)

    def _as_tensor(self, vector):
        d = self.dim
        tensor = jnp.zeros((d, d))

        # Diagonal entries
        for i in range(d):
            tensor = tensor.at[i, i].set(vector[i])

        # Off-diagonal entries (upper triangle) scaled by 1/sqrt(2)
        offset = d
        for i in range(d):
            for j in range(i + 1, d):
                val = vector[offset] / jnp.sqrt(2)
                tensor = tensor.at[i, j].set(val)
                tensor = tensor.at[j, i].set(val)  # symmetry
                offset += 1

        return tensor

    def __matmul__(self, other):
        # Multiplication of symmetric tensors cannot be ensured to remain symmetric
        return Tensor2(tensor=self.tensor @ other.tensor)
