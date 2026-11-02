import numpy as np


def clip_gradient_norm(gradient, max_grad_norm, inplace=False):
    """Cap each row's L2 norm of ``gradient`` to ``max_grad_norm``."""
    if not inplace:
        gradient = gradient.copy()
    norm = np.linalg.norm(gradient, axis=1)
    coeff = max_grad_norm / (norm + 1e-6)
    mask = coeff < 1
    gradient[mask] *= coeff[mask, None]
    return gradient


def clip_step_norm(update, max_step_norm, inplace=False):
    """Cap each row's L2 norm of ``update`` to ``max_step_norm``."""
    if not inplace:
        update = update.copy()
    norms = np.linalg.norm(update, axis=1)
    mask = norms > max_step_norm
    update[mask] *= (max_step_norm / norms[mask])[:, None]
    return update


class DeltaBarDeltaOptimizer:
    """Per-parameter delta-bar-delta ("gains") optimizer with momentum.

    This is the classic t-SNE optimizer (Jacobs, 1988): momentum gradient
    descent with a per-element adaptive learning-rate multiplier, the ``gain``.
    The effective step is

        velocity = momentum * velocity - learning_rate * gain * gradient

    and the parameter is updated by adding ``update``. The ``gain`` adapts per
    element with an additive-increase / multiplicative-decrease rule:

    - While a coordinate keeps descending in a consistent direction, its gain is
      raised by ``gain_increase`` (accelerate along a slope).
    - When a coordinate overshoots (the gradient flips to point back along the
      last step), its gain is multiplied by ``gain_decay`` (brake to settle into
      a minimum). A small ``min_gain`` floor is added so a gain never decays to
      zero, which would freeze the coordinate.

    The step moves along ``-gradient``, so while we are still descending the
    previous step (``update``) and the current ``gradient`` point in *opposite*
    directions; once we overshoot the minimum the gradient reverses and now
    *agrees in sign* with the last step. The ``flipped`` flag below is that
    overshoot condition (``sign(update) == sign(gradient)``).

    Caveat: the additive increase has no brake on a *monotone* gradient (one that
    never reverses sign), so the gain grows without bound as ``1 + gain_increase
    * iters``. This is fine for the high-dimensional embedding, whose coordinates
    oscillate and self-regulate, but pathological for a lone scalar parameter
    under a one-signed gradient. ``max_gain`` caps the gain to guard against that
    (RPROP uses the same idea); the default is unbounded, matching historical
    behavior.

    Parameters
    ----------
    gain_increase: float
        Additive gain increase per step while descending consistently (the
        historical t-SNE constant is 0.2).
    gain_decay: float
        Multiplicative gain decay applied on a sign reversal / overshoot (the
        historical t-SNE constant is 0.8).
    max_gain: float
        Upper clamp on the gain. ``inf`` (default) reproduces historical,
        uncapped behavior.

    State (``gains``, ``update``) is lazily initialized on the first step from
    the gradient's shape. Scalar parameters should be wrapped as 1-element
    ndarrays by the caller.
    """

    def __init__(self, gain_increase=0.2, gain_decay=0.8, max_gain=np.inf):
        self.gains = None
        self.update = None
        self.gain_increase = gain_increase
        self.gain_decay = gain_decay
        self.max_gain = max_gain

    def __setstate__(self, state):
        # Older pickles hard-coded the gain-schedule constants rather than
        # storing them; supply the historical defaults when absent.
        state.setdefault("gain_increase", 0.2)
        state.setdefault("gain_decay", 0.8)
        state.setdefault("max_gain", np.inf)
        self.__dict__.update(state)

    def copy(self):
        new = self.__class__(
            gain_increase=self.gain_increase,
            gain_decay=self.gain_decay,
            max_gain=self.max_gain,
        )
        if self.gains is not None:
            new.gains = np.copy(self.gains)
        if self.update is not None:
            new.update = np.copy(self.update)
        return new

    def step(self, gradient, learning_rate, momentum, min_gain=0.01):
        """Compute and return the parameter update for the current gradient.

        The caller is responsible for applying the returned update to the
        parameter. Internal ``gains`` and ``update`` state is mutated in place.
        """
        if self.update is None:
            self.update = np.zeros_like(gradient)
        if self.gains is None:
            self.gains = np.ones_like(gradient)

        # `flipped` marks coordinates that overshot: the step moves along
        # -gradient, so once we pass the minimum the gradient reverses and ends
        # up sharing a sign with the last step. Matching signs => overshoot
        # (decay the gain); differing signs => still descending (accelerate).
        flipped = np.sign(self.update) == np.sign(gradient)
        self.gains[~flipped] += self.gain_increase
        self.gains[flipped] = self.gains[flipped] * self.gain_decay + min_gain
        if np.isfinite(self.max_gain):
            np.clip(self.gains, None, self.max_gain, out=self.gains)

        self.update = momentum * self.update - learning_rate * self.gains * gradient
        return self.update

    def reset_momentum(self, mask=None):
        """Reset the per-element gains, optionally restricted to ``mask``.

        Parameters
        ----------
        mask: np.ndarray of bool or None
            Boolean array selecting which entries to reset. If ``None``,
            reset every entry.
        """
        if self.gains is None:
            return
        if mask is None:
            self.gains[:] = 0
        else:
            self.gains[mask] = 0
