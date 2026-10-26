import numpy as np
from scipy.optimize import least_squares

C = 299_792_458.0


def sanitize(h, k_idx, weights=None):
    """
    Robust Huber phase-slope estimation across OFDM subcarriers.
    Extracts direct-path phase slope m = -2*pi * df * tau, while suppressing
    subcarrier phase jumps, null tones, and dead signals.
    """
    h = np.asarray(h)
    k_idx = np.asarray(k_idx, dtype=float)

    if h.size < 4 or k_idx.size != h.size:
        return {
            "intercept": 0.0,
            "slope": 0.0,
            "sigma_fit": float("inf"),
            "coherence": 0.0,
            "valid": False,
        }

    # Amplitude check: dead signal / all zeros or NaNs
    amp = np.abs(h)
    if not np.any(np.isfinite(amp)) or np.all(amp < 1e-9):
        return {
            "intercept": 0.0,
            "slope": 0.0,
            "sigma_fit": float("inf"),
            "coherence": 0.0,
            "valid": False,
        }

    # Mask valid tones (exclude guard-band nulls <= 1e-9 and ESP32 Bin-26 LO/DC spikes > 2.25 * median)
    pos_mask = np.isfinite(amp) & (amp >= 1e-9)
    if np.sum(pos_mask) >= 8:
        med_amp = float(np.median(amp[pos_mask]))
        if med_amp > 1.0:
            valid_mask = np.isfinite(amp) & (amp > 1.0) & (amp <= 2.25 * med_amp)
            if np.sum(valid_mask) < 8:
                valid_mask = pos_mask
        else:
            valid_mask = pos_mask
    else:
        valid_mask = pos_mask
    if np.sum(valid_mask) < 4:
        return {
            "intercept": 0.0,
            "slope": 0.0,
            "sigma_fit": float("inf"),
            "coherence": 0.0,
            "valid": False,
        }

    h_v = h[valid_mask]
    k_v = k_idx[valid_mask]

    ph = np.unwrap(np.angle(h_v))
    k_centered = k_v - np.mean(k_v)
    A = np.column_stack([np.ones_like(k_centered, dtype=float), k_centered])

    if weights is not None:
        w = np.sqrt(np.asarray(weights)[valid_mask])
    else:
        amp_v = np.abs(h_v)
        med_v = float(np.median(amp_v)) + 1e-9
        w = np.clip(amp_v / med_v, 0.1, 3.0)

    def resid(p):
        return (A @ p - ph) * w

    try:
        sol = least_squares(
            resid,
            x0=[float(np.mean(ph)), 0.0],
            loss="huber",
            f_scale=0.1,
            max_nfev=25,
        )
        b, m = float(sol.x[0]), float(sol.x[1])
        r = A @ sol.x - ph
        sigma_fit = float(np.sqrt(np.mean(r**2)))
        coherence = float(np.abs(np.mean(np.exp(1j * r))))
        is_valid = bool(sol.success and coherence > 0.4 and sigma_fit < 2.5)
    except Exception:
        return {
            "intercept": 0.0,
            "slope": 0.0,
            "sigma_fit": float("inf"),
            "coherence": 0.0,
            "valid": False,
        }

    return {
        "intercept": b,
        "slope": m,
        "sigma_fit": sigma_fit,
        "coherence": coherence,
        "valid": is_valid,
    }


def slope_to_range(slope: float, df: float = 312_500.0, d_hw: float = 0.0) -> float:
    """Converts direct-path phase slope (rad/subcarrier) to distance in meters."""
    tau = -float(slope) / (2.0 * np.pi * df)
    return C * tau - d_hw


def differential_range(m_prev: float, m_now: float, df: float = 312_500.0) -> float:
    """Computes differential range increment between consecutive phase slopes."""
    return -C * (float(m_now) - float(m_prev)) / (2.0 * np.pi * df)
