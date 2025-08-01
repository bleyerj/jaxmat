from typing import Callable, Any
import jax
from jax import lax, tree_util
import jax.numpy as jnp
import lineax as lx


def newton_solve_jittable(
    f: Callable,
    x0: Any,
    args: Any,
    solver: lx.AbstractLinearSolver,
    *,
    tol: float = 1e-6,
    maxiter: int = 20,
    damping: float = 1.0,
    has_aux: bool = False,
    jac="fwd",
):
    flat_leaves, treedef = tree_util.tree_flatten(x0)
    leaf_shapes = [leaf.shape for leaf in flat_leaves]
    leaf_sizes = [leaf.size for leaf in flat_leaves]

    def vec_to_pytree(vec):
        parts = []
        idx = 0
        for shape, size in zip(leaf_shapes, leaf_sizes):
            part = vec[idx : idx + size].reshape(shape)
            parts.append(part)
            idx += size
        return tree_util.tree_unflatten(treedef, parts)

    def flat_f(vec):
        x = vec_to_pytree(vec)
        if has_aux:
            fx, aux = f(x, args)
        else:
            fx = f(x, args)
            aux = None
        flat_fx, _ = tree_util.tree_flatten(fx)
        return jnp.concatenate([jnp.ravel(leaf) for leaf in flat_fx]), aux

    fx0, aux0 = flat_f(jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves]))

    def cond_fun(state):
        i, x, fx, _, _ = state
        return jnp.logical_and(i < maxiter, jnp.linalg.norm(fx) > tol)

    def body_fun(state):
        i, x, fx, _, _ = state

        def f_no_aux(vec):
            return flat_f(vec)[0]

        if jac == "fwd":
            jacobian = jax.jacfwd
        elif jac == "bwd":
            jacobian = jax.jacrev
        J = jacobian(f_no_aux)(x)
        operator = lx.MatrixLinearOperator(J)
        solution = lx.linear_solve(operator, -fx, solver=solver)
        dx = solution.value
        x_new = x + damping * dx
        fx_new, aux = flat_f(x_new)
        return (i + 1, x_new, fx_new, dx, aux)

    x0_vec = jnp.concatenate([jnp.ravel(leaf) for leaf in flat_leaves])
    init_state = (0, x0_vec, fx0, jnp.zeros_like(x0_vec), aux0)
    _, x_final, _, _, aux_final = lax.while_loop(cond_fun, body_fun, init_state)

    result = vec_to_pytree(x_final)
    return (result, aux_final) if has_aux else result


# from jaxmat.materials import HyperelasticPotential
# from jaxmat.tensors import Tensor2, SymmetricTensor2, dev, skew, axl


# class SaintVenantKirchhoff(HyperelasticPotential):
#     mu: float
#     lmbda: float

#     def __call__(self, F):
#         E = 0.5 * (F.T @ F - SymmetricTensor2.identity())
#         return 0.5 * self.lmbda * (jnp.trace(E)) ** 2 + self.mu * jnp.trace(
#             dev(E) @ dev(E)
#         )


# material = SaintVenantKirchhoff(mu=1e3, lmbda=1e3)


# def f(F, eps):
#     PK2 = material.PK2(F)
#     res = PK2 + skew(F.inv @ PK2 - PK2.T @ (F.T).inv)
#     res = Tensor2(res.tensor.at[0, 0].set(F[0, 0] - 1 - eps))
#     return res


# def f_block(F, eps):
#     PK2 = material.PK2(F)
#     res = PK2
#     res = SymmetricTensor2(res.tensor.at[0, 0].set(F[0, 0] - 1 - eps)).array
#     res2 = axl(F.inv @ PK2 - PK2.T @ (F.T).inv)
#     return (res, res2), PK2


# # flat_x0, treedef = tree_flatten(x0)
# # x0_vec = jnp.concatenate([jnp.ravel(a) for a in flat_x0])
# # x0_ = tree_unflatten(treedef, flat_x0)
# x0 = Tensor2.identity()
# solver = lx.AutoLinearSolver(well_posed=False)
# F_sol, PK2 = newton_solve_jittable(f_block, x0, 1e-3, solver, has_aux=True)
# print(F_sol)
# print(PK2)
