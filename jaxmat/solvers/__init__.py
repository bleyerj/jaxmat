import optimistix as optx
import lineax as lx
from .custom_optimistix_solvers import (
    GaussNewtonTrustRegion,
    BFGSLinearTrustRegion,
    NewtonTrustRegion,
)

DEFAULT_LINEAR_SOLVER = lx.AutoLinearSolver(well_posed=True)
DEFAULT_SOLVERS = (
    optx.Newton(
        rtol=1e-8,
        atol=1e-8,
        linear_solver=DEFAULT_LINEAR_SOLVER,
    ),
    optx.ImplicitAdjoint(linear_solver=DEFAULT_LINEAR_SOLVER),
)
