import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import lineax as lx
from jaxmat.state import AbstractState
from jaxmat.tensors import SymmetricTensor2, dev
from jaxmat.tensors.linear_algebra import det33 as det
from .behavior import FiniteStrainBehavior
from .elasticity import LinearElasticIsotropic
from .plastic_surfaces import AbstractPlasticSurface, vonMises


def FB(x, y):
    """Scalar Fischer-Burmeister function"""
    return x + y - jnp.sqrt(x**2 + y**2)


class InternalState(AbstractState):
    p: float = eqx.field(default_factory=lambda: jnp.float64(0.0))
    be_bar: SymmetricTensor2 = SymmetricTensor2.identity()


class FeFpJ2Plasticity(FiniteStrainBehavior):
    """Material model based on https://onlinelibrary.wiley.com/doi/epdf/10.1002/nme.6843"""

    elastic_model: LinearElasticIsotropic
    yield_stress: eqx.Module
    plastic_surface: AbstractPlasticSurface = vonMises()
    internal: AbstractState = InternalState()
    # use Levenberg-Marquardt to improve convergence robustness
    solver = optx.LevenbergMarquardt(
        rtol=1e-5,
        atol=1e-5,
        linear_solver=lx.AutoLinearSolver(well_posed=False),
    )

    def constitutive_update(self, F, state, dt):
        F_old = state.F
        isv_old = state.internal
        be_bar_old = isv_old.be_bar
        p_old = isv_old.p
        Id = SymmetricTensor2.identity()

        def solve_state(F):
            # relative strain and elastic predictor
            f = F @ F_old.inv
            f_bar = f * jnp.clip(det(f), min=1e-6) ** (-1 / 3)
            be_bar_trial = f_bar.T @ be_bar_old @ f_bar

            def residual(dy, args):
                dp, be_bar = dy.p, dy.be_bar
                s = self.elastic_model.mu * dev(be_bar)
                yield_criterion = self.plastic_surface(s) - self.yield_stress(
                    p_old + dp
                )
                n = self.plastic_surface.normal(s)
                res = (
                    FB(-yield_criterion / self.elastic_model.E, dp),
                    dev(be_bar - be_bar_trial)
                    + 2 * dp * jnp.linalg.trace(be_bar) / 3 * n
                    + Id * (det(be_bar) - 1),
                )
                return res

            dy0 = isv_old.update(p=0, be_bar=be_bar_trial)
            sol = optx.root_find(residual, self.solver, dy0)
            return sol.value, be_bar_trial

        dy, be_bar_trial = solve_state(F)
        be_bar = dy.be_bar
        dp = dy.p
        y = isv_old.update(p=isv_old.p + dp, be_bar=be_bar)

        s = self.elastic_model.mu * dev(be_bar)
        J = det(F)
        tau = s + self.elastic_model.kappa / 2 * (J**2 - 1) * Id
        P = tau @ (F.T).inv

        new_state = state.update(PK1=P, internal=y)
        return P, new_state
