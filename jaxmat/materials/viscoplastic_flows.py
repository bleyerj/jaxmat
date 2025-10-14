from abc import abstractmethod
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxmat.state import make_batched


class VoceHardening(eqx.Module):
    r"""
    Voce hardening model for stress-strain behavior.

    $$
    \sigma_Y(p)=\sigma_0 + (\sigma_\text{u}-\sigma_0)(1-\exp(-bp))
    $$

    .. admonition:: References
        :class: seealso

        - Voce, E. (1955). "A Practical Strain-Hardening Function." Metallurgia, 51, 219-226.
    """

    sig0: float = eqx.field(converter=jnp.asarray)
    r"""Initial yield stress $\sigma_0$."""
    sigu: float = eqx.field(converter=jnp.asarray)
    r"""Saturation stress at large strains $\sigma_\text{u}$."""
    b: float = eqx.field(converter=jnp.asarray)
    r"""Rate of hardedning $b$."""

    def __call__(self, p):
        r"""Compute the yield stress $\sigma_Y(p)$ for a given plastic strain $p$."""
        return self.sig0 + (self.sigu - self.sig0) * (1 - jnp.exp(-1.0 * self.b * p))


class NortonFlow(eqx.Module):
    r"""A Norton viscoplastic flow with overstress.

    $$\dot{\beps}^\text{vp} = \left\langle\dfrac{f(\bsig) - \sigma_y}{K}\right\rangle_+^m$$

    where $f(\bsig)-\sigma_y$ is the overstress, $\langle \cdot\rangle_+$ is the positive part.
    """

    K: float = eqx.field(converter=jnp.asarray)
    """Characteristic stress $K$ of the Norton flow."""
    m: float = eqx.field(converter=jnp.asarray)
    """Norton power-law exponent"""

    def __call__(self, overstress):
        return jnp.maximum(overstress / self.K, 0) ** self.m


class AbstractKinematicHardening(eqx.Module):
    """An abstract module for Armstrong-Frederic type kinematic hardening."""

    nvars: eqx.AbstractVar[int]
    """The number of kinematic hardening variables"""

    @abstractmethod
    def __call__(self, X, *args):
        r"""Returns the expression for $\dot{\bX}$ as a function of the backstress $\bX$ and, possibly, other variables."""
        pass

    def sig_eff(self, sig, X):
        r"""Effective stress $\bsig-\sum_i \bX_i$ where $\bX_i$ is the $i$-th backstress."""
        return sig - jnp.sum(X, axis=0)


class LinearKinematicHardening(eqx.Module):
    r"""
    Linear kinematic hardening model.

    $$\dot{\bX} = \dfrac{2}{3}H\dot{\bepsp}$$

    .. admonition:: References
        :class: seealso

        Prager, W. (1956). A new method of analyzing stresses and strains in work-hardening plastic solids.
    """

    H: float = eqx.field(converter=jnp.asarray)
    """Linear kinematic hardening modulus"""
    nvars = 1

    @abstractmethod
    def __call__(self, eps_dot):
        r"""Returns the expression for $\dot{\bX}$ as a function of the backstress $\bX$ and, possibly, other variables."""
        return 2 / 3 * self.H * eps_dot

    def sig_eff(self, sig, X):
        r"""Effective stress $\bsig-\sum_i \bX_i$ where $\bX_i$ is the $i$-th backstress."""
        return sig - X


class ArmstrongFrederickHardening(AbstractKinematicHardening):
    r"""
    Armstrong-Frederick kinematic hardening model.

    Kinematic variables are $\ba$ such that $X_i=\frac{2}{3}C a_i$.

    .. admonition:: References
        :class: seealso

         - Armstrong, P. J., & Frederick, C. O. (1966).
            "A Mathematical Representation of the Multiaxial Bauschinger Effect for
            Hardening Materials." CEGB Report RD/B/N731.
    """

    C: jax.Array = eqx.field(converter=jnp.asarray)
    """Kinematic hardening modulus"""
    g: jax.Array = eqx.field(converter=jnp.asarray)
    """Nonlinear recall modulus"""
    nvars = 2

    def __call__(self, a, p_dot, epsp_dot):
        r"""Returns the backstress variables (in this formulation $\ba$) rate $\dot{\ba}$:

        $$\dot{\ba} = \dot{\bepsp} - g \dot{p}$$
        """
        return make_batched(epsp_dot, self.nvars) - jnp.dot(self.g, a) * p_dot

    def sig_eff(self, sig, a):
        r"""Effective stress is here:

        $$\bsig-\frac{2}{3}C\sum_{i=1}^\text{nvars}a_i$$
        """
        return sig - 2 / 3 * jnp.sum(jnp.dot(self.C, a), axis=0)
