import optimistix as optx
import lineax as lx
from .gauss_newton_ls import GaussNewtonLineSearch


DEFAULT_SOLVER = optx.Newton(
    rtol=1e-8,
    atol=1e-8,
    cauchy_termination=False,
    linear_solver=lx.AutoLinearSolver(well_posed=False),
)
