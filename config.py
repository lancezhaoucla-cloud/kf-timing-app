# config.py
from datetime import datetime
# ==========================================
# Default Application Settings
# ==========================================
DEFAULT_TICKER = "600519.sh"
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
    "ER_window": 20,              # Efficiency ratio calculation window
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
}