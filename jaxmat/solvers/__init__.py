import optimistix as optx
import lineax as lx
from .gauss_newton_ls import GaussNewtonLineSearch

linear_solver = lx.AutoLinearSolver(well_posed=False)
DEFAULT_SOLVERS = (
    optx.Newton(
        rtol=1e-8,
        atol=1e-8,
        linear_solver=linear_solver,
    ),
    optx.ImplicitAdjoint(linear_solver=linear_solver),
)
