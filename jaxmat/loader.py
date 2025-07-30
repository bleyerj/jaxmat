import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from lineax import AutoLinearSolver
from typing import Literal
from jaxmat.tensors import SymmetricTensor2


class ImposedLoading(eqx.Module):
    eps_vals: jnp.ndarray
    sig_vals: jnp.ndarray
    strain_mask: jnp.ndarray

    def __init__(
        self,
        hypothesis: Literal["small_strain", "finite_strain"] = "small_strain",
        **kwargs,
    ):
        eps_vals, sig_vals, strain_mask = _make_imposed_loading(hypothesis, **kwargs)
        self.eps_vals = eps_vals
        self.sig_vals = sig_vals
        self.strain_mask = strain_mask

    def __call__(self):
        return self.eps_vals, self.sig_vals, self.strain_mask

    def __len__(self):
        lens = {
            arr.shape[0] for arr in (self.eps_vals, self.sig_vals, self.strain_mask)
        }
        if len(lens) != 1:
            raise ValueError(
                f"Inconsistent batch sizes: {[arr.shape for arr in (self.eps_vals, self.sig_vals, self.strain_mask)]}"
            )
        return lens.pop()


def _make_imposed_loading(
    hypothesis: Literal["small_strain", "finite_strain"] = "small_strain", **kwargs
) -> ImposedLoading:

    COMPONENTS = (
        {
            f"{xi}{xj}": (i, j)
            for i, xi in enumerate("xyz")
            for j, xj in enumerate("xyz")
            # if j >= i
        }
        if hypothesis == "small_strain"
        else {
            f"{xi}{xj}": (i, j)
            for i, xi in enumerate("XYZ")
            for j, xj in enumerate("XYZ")
        }
    )
    labels = ("eps", "sig") if hypothesis == "small_strain" else ("F", "P")

    all_arrays = [jnp.atleast_1d(v) for k, v in kwargs.items()]

    if not all_arrays:
        raise ValueError("At least one array-valued argument must be provided.")

    batch_size = all_arrays[0].shape[0]
    for v in all_arrays:
        if v.shape[0] != batch_size:
            raise ValueError("All input arrays must have the same batch size.")

    eps_vals = jnp.zeros((batch_size, 3, 3))
    sig_vals = jnp.zeros((batch_size, 3, 3))
    strain_mask = jnp.zeros((batch_size, 3, 3), dtype=bool)

    # Validate keys
    valid_keys = (
        {f"{labels[0]}{k}" for k in COMPONENTS}
        | {f"{labels[1]}{k}" for k in COMPONENTS}
        | {labels[0], labels[1]}
    )

    invalid_keys = set(kwargs) - valid_keys
    if invalid_keys:
        raise ValueError(
            f"Invalid imposed loading keys: {invalid_keys}. Valid keys are {sorted(valid_keys)}."
        )

    for comp, (i, j) in COMPONENTS.items():
        e = kwargs.get(f"{labels[0]}{comp}")
        s = kwargs.get(f"{labels[1]}{comp}")
        if e is not None:
            eps_vals = eps_vals.at[:, i, j].set(e)
            strain_mask = strain_mask.at[:, i, j].set(True)
        elif s is not None:
            sig_vals = sig_vals.at[:, i, j].set(s)
            strain_mask = strain_mask.at[:, i, j].set(False)

    return eps_vals, sig_vals, strain_mask


def residual(
    material, loader: ImposedLoading, eps: jnp.ndarray, state: dict, dt: float
):
    eps_vals, sig_vals, strain_mask = loader()

    # Flatten mask to array accounting for symmetry class of strain
    if isinstance(eps, SymmetricTensor2):
        to_array = lambda x: SymmetricTensor2(tensor=x).array
    else:
        to_array = lambda x: x
    strain_mask = to_array(strain_mask)

    sig, state = material.constitutive_update(eps, state, dt)

    # Same flattening for residuals
    deps = to_array(eps - eps_vals)
    dsig = to_array(sig - sig_vals)

    eps_residual = jnp.where(strain_mask, deps, 0.0)
    sig_residual = jnp.where(jnp.logical_not(strain_mask), dsig, 0.0)
    residual_vector = jnp.where(strain_mask, eps_residual, sig_residual)
    return residual_vector, state


def stack_loadings(loadings: list):
    return jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0),
        *loadings,
    )


def solve_mechanical_state(eps0, state, loading_data: ImposedLoading, material, dt):
    solver = optx.Newton(
        rtol=1e-8, atol=1e-8, linear_solver=AutoLinearSolver(well_posed=False)
    )

    def res_fn(eps, state):
        res, new_state = residual(material, loading_data, eps, state, dt)
        return res, new_state

    sol = optx.root_find(res_fn, solver, eps0, state, has_aux=True)
    eps = sol.value
    state = sol.aux
    res, new_state = residual(material, loading_data, eps, state, dt)
    return eps, new_state, sol.stats


global_solve = jax.jit(jax.vmap(solve_mechanical_state, in_axes=(0, 0, 0, None, None)))
