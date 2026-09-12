"""Finite-horizon central response, squared gain and matched controls."""

import math

import numpy as np
import torch


def central_derivative(plus, minus, alpha):
    if not math.isfinite(alpha) or alpha <= 0 or plus.shape != minus.shape:
        raise ValueError("Positive finite alpha and matching endpoints required")
    if not torch.isfinite(plus).all() or not torch.isfinite(minus).all():
        raise ValueError("Nonfinite endpoint")
    return (plus.double() - minus.double()) / (2.0 * alpha)


def lcg_ratio_of_sums(response_norms, direction_norms):
    """sum ||central response||² / sum ||direction||², NOT a mean of ratios."""
    a, b = (
        np.asarray(response_norms, dtype=np.float64),
        np.asarray(direction_norms, dtype=np.float64),
    )
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("Matching nonempty norm vectors required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("Finite nonnegative norms required")
    denominator = float(np.square(b).sum())
    if denominator <= 0:
        raise ValueError("Zero direction energy; gain undefined")
    return float(np.square(a).sum() / denominator)


def log_contrasts(actual, shuffled, isotropic_draws, epsilon=1e-30):
    values = np.asarray([actual, shuffled, *isotropic_draws], dtype=np.float64)
    if not len(isotropic_draws) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Finite nonnegative gains and at least one isotropic draw required")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive log guard required")
    iso = float(np.mean(isotropic_draws))
    return dict(
        LCG_iso=iso,
        L_EDG=math.log((actual + epsilon) / (iso + epsilon)),
        L_EDG_state=math.log((actual + epsilon) / (shuffled + epsilon)),
        L_EDG_pop=math.log((shuffled + epsilon) / (iso + epsilon)),
        log_guard_underflow=bool((values <= epsilon).any()),
    )


@torch.inference_mode()
def finite_horizon_response(continuation, baseline, direction, alpha):
    """Two central probes; this diagnostic does not assert tangent validity.

    The continuation must be deterministic and independently resettable.
    Record realized injections: low precision need not retain requested alpha*u.
    """
    if baseline.shape != direction.shape or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Shape mismatch or invalid alpha")
    requested = direction.to(baseline.device, dtype=baseline.dtype)
    plus_state = baseline.clone() + alpha * requested
    minus_state = baseline.clone() - alpha * requested
    realized_plus = plus_state.double() - baseline.double()
    realized_minus = minus_state.double() - baseline.double()
    # Snapshot before callbacks: even an in-place continuation must not rewrite
    # the recorded injection or alias a later returned output buffer.
    plus = continuation(plus_state).detach().clone()
    minus = continuation(minus_state).detach().clone()
    derivative = central_derivative(plus, minus, alpha)
    return dict(
        derivative=derivative,
        plus=plus,
        minus=minus,
        realized_plus=realized_plus,
        realized_minus=realized_minus,
    )
