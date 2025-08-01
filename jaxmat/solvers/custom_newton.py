import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten
from jax import jacfwd, lax
import lineax as lx


def newton_solve_jittable(
    f, x0, solver: lx.AbstractLinearSolver, tol=1e-6, maxiter=20, damping=1.0
):
    flat_leaves, treedef = tree_flatten(x0)

    # Store metadata for reconstruction
    leaf_shapes = [leaf.shape for leaf in flat_leaves]
    leaf_sizes = [leaf.size for leaf in flat_leaves]
    total_size = sum(leaf_sizes)

    # Flatten x0 to a vector
    x0_vec = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])

    def vec_to_pytree(vec):
        parts = []
        idx = 0
        for shape, size in zip(leaf_shapes, leaf_sizes):
            part = vec[idx : idx + size].reshape(shape)
            parts.append(part)
            idx += size
        return tree_unflatten(treedef, parts)

    def flat_f(vec):
        x = vec_to_pytree(vec)
        fx = f(x)
        flat_fx, _ = tree_flatten(fx)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in flat_fx])

    fx0 = flat_f(x0_vec)

    def cond_fun(state):
        i, x, fx, _ = state
        return jnp.logical_and(i < maxiter, jnp.linalg.norm(fx) > tol)

    def body_fun(state):
        i, x, fx, _ = state
        J = jax.jacrev(flat_f)(x)

        operator = lx.MatrixLinearOperator(J)
        solution = lx.linear_solve(operator, -fx, solver=solver)
        dx = solution.value
        x_new = x + damping * dx
        fx_new = flat_f(x_new)
        return (i + 1, x_new, fx_new, dx)

    init_state = (0, x0_vec, fx0, jnp.zeros_like(x0_vec))
    _, x_final, _, _ = lax.while_loop(cond_fun, body_fun, init_state)

    return vec_to_pytree(x_final)


from jaxmat.materials import HyperelasticPotential
from jaxmat.tensors import Tensor2, SymmetricTensor2, dev, skew


class SaintVenantKirchhoff(HyperelasticPotential):
    mu: float
    lmbda: float

    def __call__(self, F):
        E = 0.5 * (F.T @ F - SymmetricTensor2.identity())
        return 0.5 * self.lmbda * (jnp.trace(E)) ** 2 + self.mu * jnp.trace(
            dev(E) @ dev(E)
        )


material = SaintVenantKirchhoff(mu=1e3, lmbda=1e3)


def f(F, eps):
    PK2 = material.PK2(F)
    res = PK2 + skew(F.inv @ PK2 - PK2.T @ (F.T).inv)
    res = Tensor2(res.tensor.at[0, 0].set(F[0, 0] - 1 - eps))
    return res


# flat_x0, treedef = tree_flatten(x0)
# x0_vec = jnp.concatenate([jnp.ravel(a) for a in flat_x0])
# x0_ = tree_unflatten(treedef, flat_x0)

# solver = lx.AutoLinearSolver(well_posed=False)


# F_sol = newton_solve_jittable(lambda F: f(F, 1e-3), x0, solver)
