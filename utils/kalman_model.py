import numpy as np
import pandas as pd
from scipy.optimize import minimize


class KalmanModelError(Exception):
    """Custom exception for Kalman model failures."""
    pass


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KalmanModelError(f"Missing required columns: {missing}")


def _safe_progress(progress_callback, value: float, text: str | None = None) -> None:
    if progress_callback is not None:
        progress_callback(value, text)


def kalman_fitness(theta, y_train, rvol_train, er_train, conf):
    """
    Negative log-likelihood for the Kalman filter parameters.
    Pure function: uses only passed arguments.
    """
    if len(y_train) == 0:
        raise KalmanModelError("Empty training series passed to kalman_fitness.")

    # Unpack theta safely
    c_days = float(theta[0])
    q_level = np.exp(theta[1])
    q_slope = np.exp(theta[2])
    q_cycle = np.exp(theta[3])
    r0 = np.exp(theta[4])

    # Guardrails
    c_days = max(c_days, 1e-6)

    lam = 2.0 * np.pi / c_days
    rho = float(conf.get("rho", 0.95))
    cos_l, sin_l = np.cos(lam), np.sin(lam)

    F = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, rho * cos_l, rho * sin_l],
        [0.0, 0.0, -rho * sin_l, rho * cos_l]
    ], dtype=float)

    H = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=float)
    Q_base = np.diag([q_level, q_slope, q_cycle, q_cycle]).astype(float)

    x = np.array([y_train[0], 0.0, 0.0, 0.0], dtype=float)
    P = np.eye(4, dtype=float)

    q_cap = float(conf.get("Q_scale_cap", 5.0))
    er_scale_param = float(conf.get("er_scale", 5.0))
    vol_scale_param = float(conf.get("vol_scale", 5.0))

    # --- ALIGNED: R_t Capping Logic ---
    log_returns = np.diff(y_train)
    window_volatility = float(np.std(log_returns)) if len(log_returns) > 1 else 1e-4
    max_noise_std = window_volatility * float(conf.get("max_R_t_threshold", 5.0))
    MAX_R = max(max_noise_std ** 2, 1e-10)
    MIN_R = float(conf.get("min_R_t", 1e-10))

    # --- ALIGNED: Dynamic Burn-in Period ---
    cycle_z_window = int(conf.get("cycle_z_window", 20))
    er_window = int(conf.get("ER_window", 20))
    burn_in_days = max(20, cycle_z_window, er_window)

    n_train = len(y_train)
    log_likelihood = 0.0

    for t in range(n_train):
        current_er = float(np.clip(er_train[t], 0.0, 1.0))
        q_expansion_factor = min(1.0 + (1.0 - current_er) * er_scale_param, q_cap)
        Q = Q_base * q_expansion_factor

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        r_t = y_train[t] - y_train[t - 1] if t > 0 else 0.0
        jump_score = abs(r_t) / (window_volatility + 1e-8)
        jump_factor = 1.0 + conf.get("jump_alpha", 4) * max(0.0, jump_score - conf.get("jump_threshold", 2))
        rv = max(float(rvol_train[t]), 1e-6)
        vol_scale = np.sqrt(1.0 / vol_scale_param / rv)
        
        # --- ALIGNED: Apply bounds to R_t ---
        raw_R_t = r0 * vol_scale*jump_factor
        R_t = min(max(raw_R_t, MIN_R), MAX_R)

        y_pred = float((H @ x_pred).item())
        innovation = float(y_train[t] - y_pred)
        S = float((H @ P_pred @ H.T).item() + R_t)
        S = max(S, 1e-10)

        # --- ALIGNED: Penalize only after burn-in finishes ---
        if t >= burn_in_days:
            log_likelihood += -0.5 * (
                np.log(2.0 * np.pi) + np.log(S) + (innovation ** 2) / S
            )

        K = P_pred @ H.T / S
        x = x_pred + (K.flatten() * innovation)
        P = (np.eye(4) - K @ H) @ P_pred

    return -log_likelihood


def optimize_kalman_parameters(df: pd.DataFrame, conf: dict) -> dict:
    """
    Extract arrays from the dataframe and run L-BFGS-B optimization.
    Returns a dictionary of optimized parameters.
    """
    _validate_required_columns(df, ["er", "log_close", "rvol"])

    er_values = df["er"].to_numpy(dtype=float)
    y = df["log_close"].to_numpy(dtype=float)
    rvol = df["rvol"].fillna(1.0).to_numpy(dtype=float)

    if len(y) < 30:
        raise KalmanModelError("Not enough observations for parameter optimization.")

    init_theta = [
        20.0,
        np.log(1e-4),
        np.log(1e-5),
        np.log(1e-4),
        np.log(1e-2),
    ]

    bounds = [
        (3.0, 60.0),   # cycle days
        (-16.0, -14.0),# q_level
        (-16.0, -12.0),# q_slope
        (-16.0, -10.0),# q_cycle
        (-8.0, -2.0),  # r0
    ]

    res = minimize(
        kalman_fitness,
        init_theta,
        args=(y, rvol, er_values, conf),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not res.success:
        raise KalmanModelError(f"Kalman parameter optimization failed: {res.message}")

    optimized_params = {
        "opt_cycle_days": float(res.x[0]),
        "opt_q_level": float(np.exp(res.x[1])),
        "opt_q_slope": float(np.exp(res.x[2])),
        "opt_q_cycle": float(np.exp(res.x[3])),
        "opt_r0": float(np.exp(res.x[4])),
    }

    return optimized_params


def run_kalman_filter(
    df: pd.DataFrame,
    optimized_params: dict,
    conf: dict,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Apply the Kalman filter using optimized parameters and append
    predictions, states, signal logic, and backtest results to the dataframe.
    """
    _validate_required_columns(df, ["er", "log_close", "rvol", "limit_pct", "Close"])

    df = df.copy()

    er_values = df["er"].to_numpy(dtype=float)
    y = df["log_close"].to_numpy(dtype=float)
    rvol = df["rvol"].fillna(1.0).to_numpy(dtype=float)
    limit_pct_values = df["limit_pct"].to_numpy(dtype=float)
    n = len(y)

    if n == 0:
        raise KalmanModelError("Empty dataframe passed to run_kalman_filter.")

    opt_cycle_days = float(optimized_params["opt_cycle_days"])
    opt_q_level = float(optimized_params["opt_q_level"])
    opt_q_slope = float(optimized_params["opt_q_slope"])
    opt_q_cycle = float(optimized_params["opt_q_cycle"])
    opt_r0 = float(optimized_params["opt_r0"])

    rho = float(conf.get("rho", 0.95))
    lam = 2.0 * np.pi / max(opt_cycle_days, 1e-6)
    cos_l, sin_l = np.cos(lam), np.sin(lam)

    F = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, rho * cos_l, rho * sin_l],
        [0.0, 0.0, -rho * sin_l, rho * cos_l]
    ], dtype=float)

    H = np.array([[1.0, 0.0, 1.0, 0.0]], dtype=float)

    Q_base = np.array([
        [opt_q_level, 0.0, 0.0, 0.0],
        [0.0, opt_q_slope, 0.0, 0.0],
        [0.0, 0.0, opt_q_cycle, 0.0],
        [0.0, 0.0, 0.0, opt_q_cycle]
    ], dtype=float)

    R0 = opt_r0
    x = np.array([y[0], 0.0, 0.0, 0.0], dtype=float)
    P = np.eye(4, dtype=float)

    x_filtered = np.zeros((n, 4), dtype=float)
    P_filtered = np.zeros((n, 4, 4), dtype=float)
    pred_y_current = np.full(n, np.nan, dtype=float)
    pred_y_next = np.full(n, np.nan, dtype=float)
    used_R = np.zeros(n, dtype=float)

    log_returns = np.diff(y)
    window_volatility = float(np.std(log_returns)) if len(log_returns) > 1 else 1e-4
    max_noise_std = window_volatility * float(conf.get("max_R_t_threshold", 5.0))
    MAX_R = max(max_noise_std ** 2, 1e-10)
    MIN_R = float(conf.get("min_R_t", 1e-10))

    q_cap = float(conf.get("Q_scale_cap", 5.0))
    er_scale_param = float(conf.get("er_scale", 5.0))
    vol_scale_param = float(conf.get("vol_scale", 5.0))

    for t in range(n):
        if t % 10 == 0 or t == n - 1:
            _safe_progress(progress_callback, (t + 1) / n, f"Running Kalman Filter... ({t+1}/{n})")

        current_er = float(np.clip(er_values[t], 0.0, 1.0))
        q_expansion_factor = min(1.0 + (1.0 - current_er) * er_scale_param, q_cap)
        Q = Q_base * q_expansion_factor

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        rv = max(float(rvol[t]), 1e-6)
        r_t = float(y[t] - y[t - 1]) if t > 0 else 0.0
        vol_scale = np.sqrt(1.0 / vol_scale_param / rv)
        jump_score = abs(r_t) / (window_volatility + 1e-8)
        jump_factor = 1.0 + conf.get("jump_alpha", 4) * max(0.0, jump_score - conf.get("jump_threshold", 2))
        raw_R_t = R0 * vol_scale*jump_factor
        R_t = min(max(raw_R_t, MIN_R), MAX_R)
        used_R[t] = R_t

        y_pred = float((H @ x_pred.reshape(-1, 1)).item())
        pred_y_current[t] = y_pred

        y_observed = float(y[t])

        current_limit = float(limit_pct_values[t])
        upper_log_limit = np.log(1.0 + current_limit)
        lower_log_limit = abs(np.log(1.0 - current_limit))

        is_limit_up = r_t > upper_log_limit
        is_limit_down = r_t < -lower_log_limit

        if is_limit_up or is_limit_down:
            unreleased_momentum = 1.0 / (1.0 + rv)
            shadow_premium = np.sign(r_t) * window_volatility * unreleased_momentum
            y_observed = float(y[t] + shadow_premium)

        innovation = y_observed - y_pred
        S = float((H @ P_pred @ H.T).item() + R_t)
        S = max(S, 1e-10)

        K = P_pred @ H.T / S
        x = x_pred + (K.flatten() * innovation)
        P = (np.eye(4) - K @ H) @ P_pred

        x_filtered[t] = x
        P_filtered[t] = P

        x_next_pred = F @ x
        y_next_pred = float((H @ x_next_pred.reshape(-1, 1)).item())
        pred_y_next[t] = y_next_pred

    _safe_progress(progress_callback, 1.0, "Kalman Filter complete.")

    df["kf_level"] = x_filtered[:, 0]
    df["kf_slope"] = x_filtered[:, 1]
    df["kf_cycle"] = x_filtered[:, 2]
    df["kf_cycle_aux"] = x_filtered[:, 3]
    df["used_R"] = used_R

    df["pred_log_close_current"] = pred_y_current
    df["pred_log_close_next"] = pred_y_next

    df["pred_ret_1d"] = np.exp(df["pred_log_close_next"] - df["log_close"]) - 1.0
    df["real_ret_1d"] = df["Close"].shift(-1) / df["Close"] - 1.0

    df["trend_plus_cycle"] = df["kf_level"] + df["kf_cycle"]
    df["residual"] = df["log_close"] - df["trend_plus_cycle"]

    # Signal logic
    df["trend_slope"] = df["kf_slope"]
    df["cycle_slope"] = df["kf_cycle"].diff()

    z_window = int(conf["cycle_z_window"])
    df["cycle_std"] = df["kf_cycle"].rolling(window=z_window, min_periods=z_window).std()
    df["cycle_z"] = df["kf_cycle"] / (df["cycle_std"] + 1e-6)

    trend_threshold = float(conf["trend_threshold"])
    cycle_slope_threshold = float(conf["cycle_slope_threshold"])
    cycle_z_threshold = float(conf["cycle_z_threshold"])
    cycle_snr_threshold = float(conf["cycle_snr_threshold"])

    df["recent_price_vol"] = df["log_close"].diff().rolling(window=z_window, min_periods=z_window).std()
    df["dynamic_cycle_threshold"] = df["recent_price_vol"] * cycle_snr_threshold
    df["is_cycle_alive"] = df["cycle_std"] > df["dynamic_cycle_threshold"]

    log_trend_threshold = np.log1p(trend_threshold)
    log_cycle_slope_threshold = np.log1p(cycle_slope_threshold)

    condition_with_cycle = (
        (df["trend_slope"] > log_trend_threshold) &
        (df["cycle_slope"] > log_cycle_slope_threshold) &
        (df["cycle_z"] < cycle_z_threshold) &
        ((df["pred_log_close_next"] - df["log_close"]) > 0)
    )

    condition_trend_only = df["trend_slope"] > log_trend_threshold

    long_condition = np.where(df["is_cycle_alive"], condition_with_cycle, condition_trend_only)
    df["signal_long"] = np.where(long_condition, 1, 0)

    burn_in_days = max(20, int(conf["cycle_z_window"]), int(conf["ER_window"]))
    df.loc[:burn_in_days - 1, "signal_long"] = 0

    # Backtest
    df["stock_ret_today"] = df["Close"].pct_change().fillna(0.0)
    df["position"] = df["signal_long"].shift(1).fillna(0.0)
    df["strategy_ret_today"] = df["position"] * df["stock_ret_today"]
    df["turnover"] = df["position"].diff().abs().fillna(0.0)

    slippage = float(conf["slippage"])
    df["strategy_ret_today_after_cost"] = df["strategy_ret_today"] - slippage * df["turnover"]

    df["cum_stock"] = (1.0 + df["stock_ret_today"]).cumprod()
    df["cum_strategy"] = (1.0 + df["strategy_ret_today"]).cumprod()
    df["cum_strategy_after_cost"] = (1.0 + df["strategy_ret_today_after_cost"]).cumprod()

    df["excess_vs_bh"] = df["cum_strategy"] / df["cum_stock"] - 1.0
    df["excess_vs_bh_after_cost"] = df["cum_strategy_after_cost"] / df["cum_stock"] - 1.0

    return df