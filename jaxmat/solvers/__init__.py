import optimistix as optx
import lineax as lx
from .custom_optimistix_solvers import (
    GaussNewtonTrustRegion,
    BFGSLinearTrustRegion,
    NewtonTrustRegion,
)

linear_solver = lx.AutoLinearSolver(well_posed=False)
DEFAULT_SOLVERS = (
    optx.Newton(
        rtol=1e-8,
        atol=1e-8,
        linear_solver=linear_solver,
    ),
    optx.ImplicitAdjoint(linear_solver=linear_solver),
)
