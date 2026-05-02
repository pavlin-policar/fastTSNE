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
    """Per-parameter delta-bar-delta optimizer with momentum.

    Maintains an adaptive per-element ``gain`` and a momentum-smoothed
    ``update`` for a single ndarray parameter. On every :meth:`step`, gains
    are increased where the gradient sign agrees with the previous update and
    decayed otherwise (Jacobs, 1988), then the update is computed as
    ``momentum * update - learning_rate * gain * gradient``.

    State (``gains``, ``update``) is lazily initialized on the first step from
    the gradient's shape. Scalar parameters should be wrapped as 1-element
    ndarrays by the caller.
    """

    def __init__(self):
        self.gains = None
        self.update = None

    def copy(self):
        new = self.__class__()
        if self.gains is not None:
            new.gains = np.copy(self.gains)
        if self.update is not None:
            new.update = np.copy(self.update)
        return new

    def step(self, gradient, learning_rate, momentum, min_gain=0.01):
        """Compute and return the parameter update for the current gradient.

        The caller is responsible for applying the returned update to the
        parameter. Internal ``gains`` and ``update`` state is mutated in
        place.
        """
        if self.update is None:
            self.update = np.zeros_like(gradient)
        if self.gains is None:
            self.gains = np.ones_like(gradient)

        flipped = np.sign(self.update) != np.sign(gradient)
        self.gains[flipped] += 0.2
        self.gains[~flipped] = self.gains[~flipped] * 0.8 + min_gain

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
