import jax
import jax.numpy as jnp
import equinox as eqx

COMPONENTS = ["xx", "yy", "zz", "xy", "xz", "yz"]


def create_imposed_loading(
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (eps_indices, eps_vals, sig_indices, sig_vals) as jnp arrays."""
    imposed_strains: dict[int, float] = {}
    imposed_stresses: dict[int, float] = {}

    for i, comp in enumerate(COMPONENTS):
        eij = kwargs.get(f"eps{comp}")
        sij = kwargs.get(f"sig{comp}", 0.0)
        if eij is not None:
            imposed_strains[i] = jnp.asarray(eij)
        else:
            imposed_stresses[i] = jnp.asarray(sij)

    return imposed_strains, imposed_stresses


def extract_loading_data(data):
    return jnp.array(list(data.keys()), dtype=jnp.int32), jnp.array(
        list(data.values()), dtype=jnp.float64
    )


def create_loading_residual(material):
    def residual(eps, aux):
        load_data, state, dt = aux
        imposed_eps, imposed_sig = load_data
        sig, state = material.constitutive_update(eps, state, dt)

        sig = state.stress
        eps_indices, eps_vals = extract_loading_data(imposed_eps)
        sig_indices, sig_vals = extract_loading_data(imposed_sig)

        eps_res = eps[eps_indices] - eps_vals
        sig_res = sig[sig_indices] - sig_vals
        return (
            jnp.concatenate((jnp.atleast_1d(eps_res), jnp.atleast_1d(sig_res))),
            state,
        )

    return residual


# TODO: ARC LENGTH
# def create_arclength_residual(material):
#     def residual(y, aux):
#         eps, lamb = y
#         load_data, state, dt = aux
#         imposed_eps, imposed_sig = load_data
#         sig, state = material.constitutive_update(eps, state, dt)

#         sig = state.stress
#         eps_indices, eps_vals = extract_loading_data(imposed_eps)
#         sig_indices, sig_vals = extract_loading_data(imposed_sig)

#         eps_res = eps[eps_indices] - lamb*eps_vals
#         sig_res = sig[sig_indices] - lamb*sig_vals
#         lamb_res =
#         return (
#             jnp.concatenate((jnp.atleast_1d(eps_res), jnp.atleast_1d(sig_res))),
#             state,
#         )

#     return residual
