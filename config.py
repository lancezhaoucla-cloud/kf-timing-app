# config.py
from datetime import datetime
# ==========================================
# Default Application Settings
# ==========================================
DEFAULT_TICKER = "300750.sz"
DEFAULT_END_DATE = datetime.today().strftime('%Y%m%d')

# ==========================================
# Kalman Filter & Feature Hyperparameters
# ==========================================
KALMAN_PARAMS = {
    "rolling_window": 250,
    "trend_threshold": 0.0,
    "cycle_slope_threshold": 0.0,
    "cycle_z_threshold": 3,
    "cycle_z_window": 20,         # Minimum 20 days for peak judgment
    "ER_window": 20,
    "volume_window": 120,             # Relative volume calculation window
    "slippage": 0.000,
    "vol_scale": 10,
    "er_scale": 10,
    "cycle_regime_threshold": 0.02, # Currently unused, reserved for future expansion
    "min_R_t": 1e-6,
    "max_R_t_threshold": 10,
    "return_limit": 0.095,        # Default limit, dynamically overridden by API
    "cycle_snr_threshold": 0.8,   # Dynamic regime switch threshold
    "Q_scale_cap": 10,
    "jump_threshold": 2,
    "jump_alpha": 4,

        # --- Classifier Parameters (profile-INDEPENDENT) ---
    "classifier_window": 20,               # Main lookback for efficiency / coherence / curvature
    "breakout_return_window": 3,           # Short return horizon used for ignition
    "breakout_reference_window": 20,       # Reference lookback for return shock normalization
    "breakout_volume_short_window": 3,     # Recent average volume window
    "breakout_volume_long_window": 20,     # Baseline average volume window
    "breakout_return_z_threshold": 1.5,    # |recent_k_ret - mean(hist_k_ret)| / std(hist_k_ret)
    "breakout_volume_ratio_threshold": 1.5,# recent short vol / long vol
    "breakout_min_tradability": 0.30,

    # --- Regime Routing Thresholds ---
    "tradability_hi": 0.60,
    "tradability_lo": 0.30,
    "defender_tradability": 0.1,

    # Curvature is scale-dependent, so normalize it before thresholding
    "transition_z_threshold": 1.5,

    # --- Simple Hysteresis ---
    "trend_hysteresis_factor": 0.90,       # If already in trend, tolerate some deterioration
    "breakout_hold_if_ignite": True,
}

PROFILE_CONFIGS = {
    "Trend_follower": {
        "trend_threshold": 0.0,
        "cycle_slope_threshold": 0.0,
        "cycle_z_threshold": 4.0,
        "vol_scale": 1.0,
        "er_scale": 10.0,
        "cycle_snr_threshold": 1.0,
        "jump_threshold": 2.0,
        "jump_alpha": 4.0,
        "max_R_t_threshold": 10.0,
        "Q_scale_cap": 3.0,

        "volume_window": 20,
        "ER_window": 20,
        "rho": 0.97,
    },
    "Breakseeker": {
        "trend_threshold": 0.0,
        "cycle_slope_threshold": -0.001,
        "cycle_z_threshold": 10.0,
        "vol_scale": 10.0,
        "er_scale": 1.0,
        "cycle_snr_threshold": 0.8,
        "jump_threshold": 2.5,
        "jump_alpha": 0.5,
        "max_R_t_threshold": 3.0,
        "Q_scale_cap": 10.0,

        "volume_window": 120,
        "ER_window": 20,
        "rho": 0.90,
    },
    "Defender": {
        "trend_threshold": 0.001,
        "cycle_slope_threshold": 0.001,
        "cycle_z_threshold": 2.5,
        "vol_scale": 5.0,
        "er_scale": 5.0,
        "cycle_snr_threshold": 0.2,
        "jump_threshold": 2.0,
        "jump_alpha": 2.0,
        "max_R_t_threshold": 5.0,
        "Q_scale_cap": 5.0,

        "volume_window": 20,
        "ER_window": 20,
        "rho": 0.95,
    },
    "Activist": {
        "trend_threshold": 0.0,
        "cycle_slope_threshold": 0.0,
        "cycle_z_threshold": 3.0,
        "vol_scale": 10.0,
        "er_scale": 10.0,
        "cycle_snr_threshold": 0.5,
        "jump_threshold": 2.0,
        "jump_alpha": 1.0,
        "max_R_t_threshold": 10.0,
        "Q_scale_cap": 10.0,

        "volume_window": 20,
        "ER_window": 20,
        "rho": 0.95,
    },
    "All_other": {
        "trend_threshold": 0.0,
        "cycle_slope_threshold": 0.0,
        "cycle_z_threshold": 3.0,
        "vol_scale": 3.0,
        "er_scale": 3.0,
        "cycle_snr_threshold": 0.5,
        "jump_threshold": 2.0,
        "jump_alpha": 2.0,
        "max_R_t_threshold": 5.0,
        "Q_scale_cap": 5.0,

        "volume_window": 20,
        "ER_window": 20,
        "rho": 0.95,
    },
}

def build_profile_config(base_config, profile_name):
    """Merge BASE_CONFIG with a selected profile override."""
    conf = base_config.copy()
    conf.update(PROFILE_CONFIGS[profile_name])
    return conf