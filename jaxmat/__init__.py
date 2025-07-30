import jax

jax.config.update("jax_enable_x64", True)  # use double-precision
jax.config.update("jax_debug_nans", True)  # raise when encountering nan

from jaxmat.solvers.gauss_newton_ls import GaussNewtonLineSearch
