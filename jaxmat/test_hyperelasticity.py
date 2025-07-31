import matplotlib.pyplot as plt
import jax.numpy as jnp

from jaxmat.loader import ImposedLoading, global_solve
from jaxmat.materials.hyperelasticity import (
    Hyperelasticity,
    CompressibleGhentMooneyRivlin,
    CompressibleNeoHookean,
)


def test_hyperelasticity():

    # material = Hyperelasticity(CompressibleNeoHookean(mu=2.8977, kappa=1e3))
    material = Hyperelasticity(
        CompressibleGhentMooneyRivlin(c1=2.8977, c2=0.0635, Jm=91.41, kappa=1e3)
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
        F_sol, state_sol, stats = global_solve(F0, state, loading, material, dt)

        F.append(F_sol)
        Sig.append(state_sol.stress)

    for f, sig in zip(F, Sig):
        plt.plot(f[:, 0, 0], sig[:, 0, 0], "x-")

    traction_simple = [
        (1.00, 0),
        (1.36, 3.16),
        (1.60, 4.27),
        (1.89, 5.45),
        (2.16, 6.13),
        (2.42, 6.99),
        (3.07, 8.97),
        (3.59, 10.64),
        (4.03, 12.31),
        (4.79, 16.27),
        (5.37, 20.05),
        (5.84, 23.39),
        (6.17, 27.36),
        (6.43, 30.77),
        (6.69, 34.42),
        (6.93, 38.08),
        (7.06, 42.04),
        (7.19, 45.56),
        (7.34, 49.23),
        (7.47, 53.02),
        (7.58, 56.73),
        (7.65, 64.30),
    ]
    traction_plane = [
        (1.00, 0),
        (1.06, 0.80),
        (1.10, 1.55),
        (1.16, 2.17),
        (1.26, 2.79),
        (1.40, 4.03),
        (1.83, 5.88),
        (2.40, 7.55),
        (3.00, 9.53),
        (3.51, 10.95),
        (4.05, 12.68),
        (4.45, 14.66),
        (5.06, 18.00),
    ]
    traction_equibiaxiale = [
        (1.00, 0),
        (1.04, 2.17),
        (1.17, 3.35),
        (1.29, 4.40),
        (1.34, 4.83),
        (1.43, 5.39),
        (1.66, 6.50),
        (1.90, 7.80),
        (2.49, 9.78),
        (3.01, 12.82),
        (3.39, 14.67),
        (3.73, 17.27),
        (4.08, 20.31),
        (4.24, 22.17),
        (4.47, 24.96),
    ]

    # Séparer λ₁ et t pour chaque essai
    lambda_uni_exp, t_uni_exp = zip(*traction_simple)
    lambda_plane_exp, t_plane_exp = zip(*traction_plane)
    lambda_equi_exp, t_equi_exp = zip(*traction_equibiaxiale)

    plt.plot(lambda_uni_exp, t_uni_exp, label="Traction simple", marker="o")
    plt.plot(lambda_plane_exp, t_plane_exp, label="Traction plane", marker="s")
    plt.plot(lambda_equi_exp, t_equi_exp, label="Traction équibiaxiale", marker="^")
    # plt.legend()
    plt.show()


test_hyperelasticity()
