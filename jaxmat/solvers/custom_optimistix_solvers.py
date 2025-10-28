from collections.abc import Callable
from jaxtyping import PyTree, Scalar
import optimistix as optx
import lineax as lx
from optimistix import max_norm


class GaussNewtonTrustRegion(optx.GaussNewton):
    """Gauss-Newton algorithm, for solving nonlinear least-squares problems with a trust region search."""

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = optx.NewtonDescent(linear_solver=linear_solver)
        self.search = optx.ClassicalTrustRegion()
        self.verbose = verbose


class NewtonTrustRegion(optx.LevenbergMarquardt):
    """Newton method for solving nonlinear problems with a trust region search.

    Notes
    -----
    This algorithm is derived from LevenbergMarquardt using a full Newton descent
    instead of a damped one.
    """

    def __init__(
        self,
        rtol: float,
        atol: float,
        norm: Callable[[PyTree], Scalar] = max_norm,
        linear_solver: lx.AbstractLinearSolver = lx.AutoLinearSolver(well_posed=None),
        verbose: frozenset[str] = frozenset(),
    ):
        self.rtol = rtol
        self.atol = atol
        self.norm = norm
        self.descent = optx.NewtonDescent(linear_solver=linear_solver)
        self.search = optx.ClassicalTrustRegion()
        self.verbose = verbose


class BFGSLinearTrustRegion(optx.AbstractBFGS):
    """Standard BFGS algorithm, for solving minimisation problems with a linear trust region search."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.LinearTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()
