from collections.abc import Callable
from jaxtyping import PyTree, Scalar
import optimistix as optx
import lineax as lx
from optimistix import max_norm


class GaussNewtonLineSearch(optx.GaussNewton):
    """Gauss-Newton algorithm, for solving nonlinear least-squares problems.

    Note that regularised approaches like [`optimistix.LevenbergMarquardt`][] are
    usually preferred instead.

    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
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


class LevenbergMarquardtLineSearch(optx.LevenbergMarquardt):
    """Levenberg-Marquardt algorithm, for solving nonlinear least-squares problems.


    Supports the following `options`:

    - `jac`: whether to use forward- or reverse-mode autodifferentiation to compute the
        Jacobian. Can be either `"fwd"` or `"bwd"`. Defaults to `"fwd"`, which is
        usually more efficient. Changing this can be useful when the target function has
        a `jax.custom_vjp`, and so does not support forward-mode autodifferentiation.
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
    """Standard BFGS + linear trust region update."""

    rtol: float
    atol: float
    norm: Callable = optx.max_norm
    use_inverse: bool = True
    search: optx.AbstractSearch = optx.ClassicalTrustRegion()
    descent: optx.AbstractDescent = optx.NewtonDescent()
    verbose: frozenset[str] = frozenset()
