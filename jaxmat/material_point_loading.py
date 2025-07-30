import jax
import jax.numpy as jnp
import equinox as eqx

# ss_COMPONENTS = ["xx", "yy", "zz", "xy", "xz", "yz"]
# fs_COMPONENTS = ["XX", "YY", "ZZ", "XY", "YX", "XZ", "ZX", "YZ", "ZY"]
# fs_COMPONENTS = [["XX", "XY", "XZ"], ["YX", "YY", "YZ"], ["ZX", "ZY", "ZZ"]]
ss_COMPONENTS = {
    f"{xi}{xj}": (i, j)
    for i, xi in enumerate(["x", "y", "z"])
    for j, xj in enumerate(["x", "y", "z"])
    if j >= i
}
fs_COMPONENTS = {
    f"{xi}{xj}": (i, j)
    for i, xi in enumerate(["X", "Y", "Z"])
    for j, xj in enumerate(["X", "Y", "Z"])
}


def create_imposed_loading(
    type="small_strain",
    **kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (eps_indices, eps_vals, sig_indices, sig_vals) as jnp arrays."""
    imposed_strains: dict[int, float] = {}
    imposed_stresses: dict[int, float] = {}
    if type == "small_strain":
        COMPONENTS = ss_COMPONENTS
        labels = ("eps", "sig")
    elif type == "finite_strain":
        COMPONENTS = fs_COMPONENTS
        labels = ("F", "P")
    else:
        raise ValueError("Only `small_strain` or `finite_strain` is supported")
    # for i, comp in enumerate(COMPONENTS):
    for comp, i in COMPONENTS.items():
        eij = kwargs.get(f"{labels[0]}{comp}")
        sij = kwargs.get(f"{labels[1]}{comp}", 0.0)
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

        eps_res = jnp.asarray(
            [eps[*epsi] - epsv for epsi, epsv in zip(eps_indices, eps_vals)]
        )
        sig_res = jnp.asarray(
            [sig[*sigi] - sigv for sigi, sigv in zip(sig_indices, sig_vals)]
        )
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
