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
    """Ratio of squared-norm sums, with a scaled fallback at float64 extremes.

    A true zero denominator or an unrepresentable positive ratio raises ValueError.
    Ordinary-range inputs retain the original sum/divide arithmetic.
    """
    a, b = (
        np.asarray(response_norms, dtype=np.float64),
        np.asarray(direction_norms, dtype=np.float64),
    )
    if a.shape != b.shape or a.ndim != 1 or not a.size:
        raise ValueError("Matching nonempty norm vectors required")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a < 0).any() or (b < 0).any():
        raise ValueError("Finite nonnegative norms required")
    amax, bmax = float(a.max()), float(b.max())
    if bmax == 0:
        raise ValueError("Zero direction energy; gain undefined")
    if amax == 0:
        return 0.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        aa, bb = np.square(a), np.square(b)
        numerator, denominator = float(aa.sum()), float(bb.sum())
        ratio = float(np.divide(numerator, denominator))
    tiny = np.finfo(np.float64).tiny
    lost_range = ((a > 0) & (aa < tiny)).any() or ((b > 0) & (bb < tiny)).any()
    if not lost_range and math.isfinite(numerator) and math.isfinite(denominator):
        if math.isfinite(ratio) and ratio > 0:
            return ratio
    # Separate mantissa/exponent scales avoid squaring extreme absolute norms.
    with np.errstate(under="ignore"):
        energy_ratio = float(np.square(a / amax).sum() / np.square(b / bmax).sum())
    ma, ea = math.frexp(amax)
    mb, eb = math.frexp(bmax)
    try:
        ratio = math.ldexp((ma / mb) ** 2 * energy_ratio, 2 * (ea - eb))
    except OverflowError as exc:
        raise ValueError("Gain is outside the representable float64 range") from exc
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Gain is outside the representable float64 range")
    return ratio


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
