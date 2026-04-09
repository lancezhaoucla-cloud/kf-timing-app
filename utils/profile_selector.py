# utils/profile_selector.py

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_profile_dependent_series(df_in: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Compute profile-dependent rvol / ER series.
    These are intentionally separated from classifier logic.

    Required columns:
    - Close
    - Volume

    Optional:
    - log_close
    """
    out = pd.DataFrame(index=df_in.index)

    work = df_in.copy()

    if "log_close" not in work.columns:
        work["log_close"] = np.log(work["Close"])

    # --- Relative Volume (profile-dependent) ---
    volume_window = int(config["volume_window"])
    vol_ma = work["Volume"].rolling(volume_window).mean()
    rvol = work["Volume"] / vol_ma
    rvol = rvol.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # --- Efficiency Ratio in log space (profile-dependent) ---
    er_window = int(config["ER_window"])
    log_close = work["log_close"]

    direction = log_close.diff(er_window).abs()
    daily_diff_abs = log_close.diff().abs()
    volatility = daily_diff_abs.rolling(er_window).sum()

    er = direction / volatility
    er = er.replace([np.inf, -np.inf], np.nan).fillna(0.5)

    out["rvol"] = rvol
    out["er"] = er
    return out


def _safe_std(x, ddof: int = 1, fallback: float = 1e-8) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) <= ddof:
        return fallback
    s = np.std(x, ddof=ddof)
    if not np.isfinite(s) or s < fallback:
        return fallback
    return float(s)


def _compute_efficiency(y_cls: np.ndarray) -> float:
    """
    Direction-agnostic path efficiency:
    smooth / persistent path -> high
    back-and-forth path -> low
    """
    total_move = np.sum(np.abs(np.diff(y_cls))) + 1e-8
    return float(abs(y_cls[-1] - y_cls[0]) / total_move)


def _compute_coherence_and_curvature(y_cls: np.ndarray) -> tuple[float, float]:
    """
    Coherence: linear fit R^2
    Curvature: quadratic coefficient on normalized x
    """
    x = np.linspace(-1.0, 1.0, len(y_cls))

    # Linear fit
    p1 = np.polyfit(x, y_cls, 1)
    trend_line = np.polyval(p1, x)

    ss_res = np.sum((y_cls - trend_line) ** 2)
    ss_tot = np.sum((y_cls - np.mean(y_cls)) ** 2) + 1e-8
    coherence = float(1.0 - ss_res / ss_tot)

    # Quadratic fit
    p2 = np.polyfit(x, y_cls, 2)
    curvature = float(p2[0])

    return coherence, curvature


def _compute_breakout_ignition(y_win: np.ndarray, vol_win: np.ndarray, config: dict) -> tuple[bool, dict]:
    """
    Explicit breakout ignition using:
    1) return shock
    2) volume shock

    No direction is used for routing:
    we use |return shock|, and let KF handle sign / directional response.
    """
    k_ret = int(config["breakout_return_window"])
    ref_win = int(config["breakout_reference_window"])
    vol_short = int(config["breakout_volume_short_window"])
    vol_long = int(config["breakout_volume_long_window"])

    min_needed = max(ref_win + k_ret + 1, vol_long, vol_short + 1)
    if len(y_win) < min_needed or len(vol_win) < min_needed:
        return False, {
            "recent_k_return": np.nan,
            "return_shock_z": np.nan,
            "recent_vol_short": np.nan,
            "baseline_vol_long": np.nan,
            "volume_shock_ratio": np.nan,
            "ret_shock_on": False,
            "vol_shock_on": False,
        }

    # --- Return shock ---
    recent_k_return = float(y_win[-1] - y_win[-1 - k_ret])

    y_ref = y_win[-(ref_win + k_ret + 1):]
    hist_k_returns = y_ref[k_ret:] - y_ref[:-k_ret]

    # Exclude the most recent k-return from reference stats
    hist_base = hist_k_returns[:-1] if len(hist_k_returns) > 1 else hist_k_returns
    hist_mu = float(np.mean(hist_base)) if len(hist_base) > 0 else 0.0
    hist_sd = _safe_std(hist_base, ddof=1, fallback=1e-8)

    return_shock_z = float((recent_k_return - hist_mu) / hist_sd)
    ret_shock_on = abs(return_shock_z) >= float(config["breakout_return_z_threshold"])

    # --- Volume shock ---
    recent_vol_short = float(np.mean(vol_win[-vol_short:]))
    baseline_vol_long = float(np.mean(vol_win[-vol_long:])) + 1e-8
    volume_shock_ratio = float(recent_vol_short / baseline_vol_long)
    vol_shock_on = volume_shock_ratio >= float(config["breakout_volume_ratio_threshold"])

    breakout_ignite = bool(ret_shock_on and vol_shock_on)

    metrics = {
        "recent_k_return": recent_k_return,
        "return_shock_z": return_shock_z,
        "recent_vol_short": recent_vol_short,
        "baseline_vol_long": baseline_vol_long,
        "volume_shock_ratio": volume_shock_ratio,
        "ret_shock_on": ret_shock_on,
        "vol_shock_on": vol_shock_on,
    }
    return breakout_ignite, metrics


def compute_classifier_features(y_win: np.ndarray, vol_win: np.ndarray, config: dict) -> dict:
    """
    Classifier-only feature block.
    Explicitly independent from profile-specific ER / volume windows.
    """
    cls_win = int(config["classifier_window"])

    min_needed = max(
        cls_win,
        int(config["breakout_reference_window"]) + int(config["breakout_return_window"]) + 1,
        int(config["breakout_volume_long_window"]),
    )

    if len(y_win) < min_needed or len(vol_win) < min_needed:
        return {
            "efficiency": np.nan,
            "coherence": np.nan,
            "curvature": np.nan,
            "curvature_z": np.nan,
            "tradability": np.nan,
            "transition": np.nan,
            "breakout_ignite": False,
            "recent_k_return": np.nan,
            "return_shock_z": np.nan,
            "recent_vol_short": np.nan,
            "baseline_vol_long": np.nan,
            "volume_shock_ratio": np.nan,
            "ret_shock_on": False,
            "vol_shock_on": False,
        }

    y_cls = np.asarray(y_win[-cls_win:], dtype=float)

    efficiency = _compute_efficiency(y_cls)
    coherence, curvature = _compute_coherence_and_curvature(y_cls)

    # Normalize curvature by local 1-day return std to reduce scale instability
    local_ret_std = _safe_std(np.diff(y_cls), ddof=1, fallback=1e-8)
    curvature_z = float(abs(curvature) / local_ret_std)

    tradability = float(0.65 * efficiency + 0.35 * coherence)
    transition = curvature_z

    breakout_ignite, breakout_metrics = _compute_breakout_ignition(
        y_win=np.asarray(y_win, dtype=float),
        vol_win=np.asarray(vol_win, dtype=float),
        config=config,
    )

    features = {
        "efficiency": efficiency,
        "coherence": coherence,
        "curvature": curvature,
        "curvature_z": curvature_z,
        "tradability": tradability,
        "transition": transition,
        "breakout_ignite": breakout_ignite,
    }
    features.update(breakout_metrics)
    return features


def detect_profile(
    y_win: np.ndarray,
    vol_win: np.ndarray,
    prev_regime: str = "All_other",
    config: dict | None = None,
) -> tuple[str, dict]:
    """
    Upgraded profile selector.

    Parameters
    ----------
    y_win : np.ndarray
        log price window
    vol_win : np.ndarray
        raw volume window (NOT rvol)
    prev_regime : str
        previous selected profile, used for simple hysteresis
    config : dict
        classifier/base config

    Returns
    -------
    (profile_name, state_metrics)
    """
    if config is None:
        raise ValueError("detect_profile requires a config dict.")

    cfg = config.copy()
    feat = compute_classifier_features(y_win=y_win, vol_win=vol_win, config=cfg)

    # Early fallback
    if not np.isfinite(feat["tradability"]) or not np.isfinite(feat["transition"]):
        return "All_other", feat

    tradability = feat["tradability"]
    transition = feat["transition"]
    breakout_ignite = feat["breakout_ignite"]

    tradability_hi = float(cfg["tradability_hi"])
    tradability_lo = float(cfg["tradability_lo"])
    defender_tradability = float(cfg["defender_tradability"])
    breakout_min_tradability = float(cfg["breakout_min_tradability"])
    transition_z_threshold = float(cfg["transition_z_threshold"])

    # =========================
    # Main Routing
    # =========================
    if breakout_ignite and tradability >= breakout_min_tradability:
        new_profile = "Breakseeker"

    elif tradability >= tradability_hi and transition < transition_z_threshold:
        new_profile = "Trend_follower"

    elif tradability >= tradability_lo and transition >= transition_z_threshold:
        new_profile = "Activist"

    elif tradability < defender_tradability:
        new_profile = "Defender"

    else:
        new_profile = "All_other"

    # =========================
    # Simple Hysteresis
    # =========================
    if prev_regime is not None:
        # Avoid being kicked out of Trend_follower too easily
        if prev_regime == "Trend_follower" and new_profile == "All_other":
            if tradability >= tradability_hi * float(cfg["trend_hysteresis_factor"]):
                new_profile = prev_regime

        # If breakout ignition is still on, allow Breakseeker to persist briefly
        if (
            prev_regime == "Breakseeker"
            and new_profile == "All_other"
            and bool(cfg["breakout_hold_if_ignite"])
            and breakout_ignite
        ):
            new_profile = prev_regime

    return new_profile, feat


def detect_profile_from_df(
    df: pd.DataFrame,
    config: dict,
    prev_regime: str = "All_other",
) -> tuple[str, dict]:
    """
    Convenience wrapper for app usage.

    Required columns:
    - Close
    - Volume

    Optional:
    - log_close

    This function does NOT use profile-specific rvol/ER.
    It uses classifier-only logic exactly as intended.
    """
    work = df.copy().reset_index(drop=True)

    if "log_close" not in work.columns:
        work["log_close"] = np.log(work["Close"])

    y_win = work["log_close"].values.astype(float)
    vol_win = work["Volume"].values.astype(float)

    return detect_profile(
        y_win=y_win,
        vol_win=vol_win,
        prev_regime=prev_regime,
        config=config,
    )


def build_profile_debug_table(feat: dict) -> pd.DataFrame:
    """
    Optional helper for Streamlit display.
    """
    rows = []
    for k, v in feat.items():
        rows.append({"metric": k, "value": v})
    return pd.DataFrame(rows)