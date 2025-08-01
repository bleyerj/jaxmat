import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.loader import ImposedLoading, global_solve
import jaxmat.materials as jm


def test_hyperelasticity():
    # material = jm.Hyperelasticity(jm.CompressibleNeoHookean(mu=2.8977, kappa=1e3))
    material = jm.Hyperelasticity(
        jm.CompressibleGhentMooneyRivlin(c1=2.8977, c2=0.0635, Jm=91.41, kappa=1e3)
    )

    lamb_list = jnp.linspace(1.0, 8.0, 51)
    loading_uni = ImposedLoading("finite_strain", FXX=lamb_list)

    lamb_list = jnp.linspace(1.0, 8.0, 51)
    loading_simple = ImposedLoading(
        "finite_strain", FXX=lamb_list, FYY=jnp.ones_like(lamb_list)
    )

    lamb_list = jnp.linspace(1.0, 6, 51)
    loading_equiax = ImposedLoading("finite_strain", FXX=lamb_list, FYY=lamb_list)
    loadings = [loading_uni, loading_simple, loading_equiax]

    Sig = []
    F = []
    for loading in loadings:  # [loading_equiax]:
        Nbatch = len(lamb_list)
        state = material.get_state(Nbatch)
        F0 = state.F
        dt = 0.0
        F_sol, state_sol, _ = global_solve(F0, state, loading, material, dt)

        F.append(F_sol)
        Sig.append(state_sol.PK1)
        print(state_sol.PK1, state_sol)

    for f, sig in zip(F, Sig):
        plt.plot(f[:, 0, 0], sig[:, 0, 0], "x-")

    import numpy as np
    from pathlib import Path

    current_path = Path().resolve()

    data = np.loadtxt(
        current_path / "demos/identification/Treloar_data.csv",
        skiprows=1,
        delimiter=",",
    )

    plt.plot(data[:, 0], data[:, 1], label="Simple tension", marker="o")
    plt.plot(data[:, 2], data[:, 3], label="Plane tension", marker="s")
    plt.plot(data[:, 4], data[:, 5], label="Equibiaxial tension", marker="^")
    # plt.legend()
    plt.show()


test_hyperelasticity()
