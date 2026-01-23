"""Task 1: Short-term passenger flow prediction (baseline + regime AR(1)).

Functions:
- build_5min_series(csv_path): returns DataFrame indexed by interval start with columns ['count','w','tau']
- estimate_baseline_and_phi(series_df): returns mu_w (dict of arrays length 288), phi_w (dict)
- predict_next(last_ts, last_count, mu_w, phi_w): returns predicted count for next 5-min interval

Implements the simple approach described in the tentative report.
"""
from __future__ import annotations
import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd

N_INTERVALS_PER_DAY = 24 * 12  # 5-minute intervals


def build_5min_series(csv_path: str) -> pd.DataFrame:
    """Load hall calls CSV and aggregate into consecutive 5-minute intervals.

    Returns a DataFrame with index=interval_start (Timestamp), and columns:
    - count: number of hall calls in the interval
    - w: regime indicator (1=weekday, 0=weekend)
    - tau: position within day [0..287]
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # detect time column
    time_col = None
    for c in df.columns:
        if c.lower() in ("time", "timestamp", "datetime") or "time" in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError("No time column found in CSV")

    df[time_col] = pd.to_datetime(df[time_col])
    # floor to 5-minute bin (start of bin)
    df['interval_start'] = df[time_col].dt.floor('5T')
    counts = df.groupby('interval_start').size().rename('count')

    # build full index from min to max interval
    idx = pd.date_range(start=counts.index.min(), end=counts.index.max(), freq='5T')
    s = counts.reindex(idx, fill_value=0)
    out = pd.DataFrame({'count': s})
    out.index.name = 'interval_start'

    # regime w: weekday->1, weekend->0
    out['w'] = out.index.to_series().dt.weekday.apply(lambda d: 1 if d < 5 else 0).astype(int).values
    # tau: index within the day (0..N_INTERVALS_PER_DAY-1)
    out['tau'] = ((out.index.hour * 60 + out.index.minute) // 5).astype(int)

    return out


def estimate_baseline_and_phi(series_df: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    """Estimate mu_w(tau) baseline and AR(1) parameter phi for each regime w in {0,1}.

    Returns:
      mu_w: {w: array(length 288)} - baseline mean counts per 5-min slot
      phi_w: {w: float} - AR(1) coefficient estimated by least squares on deviations
    """
    # Sanity checks
    if 'count' not in series_df.columns or 'w' not in series_df.columns or 'tau' not in series_df.columns:
        raise ValueError('series_df must have count,w,tau columns')

    # Estimate mu_w(tau)
    mu_w = {}
    for w in (0, 1):
        # For each tau, average counts where w matches
        mu = np.zeros(N_INTERVALS_PER_DAY, dtype=float)
        for tau in range(N_INTERVALS_PER_DAY):
            mask = (series_df['w'].values == w) & (series_df['tau'].values == tau)
            vals = series_df['count'].values[mask]
            mu[tau] = vals.mean() if len(vals) > 0 else 0.0
        mu_w[w] = mu

    # Compute deviations e_t = y_t - mu_w(tau)
    y = series_df['count'].values
    w_arr = series_df['w'].values
    tau_arr = series_df['tau'].values
    mu_vals = np.array([mu_w[int(w)][int(tau)] for w, tau in zip(w_arr, tau_arr)])
    e = y - mu_vals

    # Estimate phi for each regime using pairs where both t and t-1 have same regime
    phi_w = {}
    # prepare previous e (lag1)
    e_prev = np.roll(e, 1)
    w_prev = np.roll(w_arr, 1)
    # For t=0, invalid; mask out t=0
    valid = np.ones_like(e, dtype=bool)
    valid[0] = False

    for w in (0, 1):
        mask = valid & (w_arr == w) & (w_prev == w)
        if mask.sum() == 0:
            phi = 0.0
        else:
            x = e_prev[mask]
            yv = e[mask]
            # OLS slope without intercept: phi = (x^T y) / (x^T x)
            denom = (x * x).sum()
            if denom == 0:
                phi = 0.0
            else:
                phi = (x * yv).sum() / denom
        phi_w[w] = float(phi)

    return mu_w, phi_w


def predict_next(last_ts: pd.Timestamp, last_count: int, mu_w: Dict[int, np.ndarray], phi_w: Dict[int, float]) -> Tuple[pd.Timestamp, float]:
    """Predict the total number of hall calls in the next 5-minute interval.

    Formula from the report:
      y_hat_{t+1} = mu_{w_{t+1}}(tau(t+1)) + phi_{w_t} * [y_t - mu_{w_t}(tau(t))]

    Inputs:
      last_ts: timestamp at start of last 5-min interval (pd.Timestamp)
      last_count: observed count y_t for that interval
      mu_w, phi_w: outputs from estimate_baseline_and_phi

    Returns: (next_interval_ts, y_hat)
    """
    if not isinstance(last_ts, pd.Timestamp):
        last_ts = pd.to_datetime(last_ts)

    next_ts = last_ts + pd.Timedelta(minutes=5)
    tau_t = ((last_ts.hour * 60 + last_ts.minute) // 5) % N_INTERVALS_PER_DAY
    tau_tp1 = ((next_ts.hour * 60 + next_ts.minute) // 5) % N_INTERVALS_PER_DAY
    w_t = 1 if last_ts.weekday() < 5 else 0
    w_tp1 = 1 if next_ts.weekday() < 5 else 0

    mu_t = mu_w[int(w_t)][int(tau_t)]
    mu_tp1 = mu_w[int(w_tp1)][int(tau_tp1)]
    phi = phi_w.get(int(w_t), 0.0)

    y_hat = mu_tp1 + phi * (last_count - mu_t)
    # predictions should not be negative
    y_hat = max(0.0, float(y_hat))
    return next_ts, y_hat


if __name__ == '__main__':
    # Quick local smoke test if run directly
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.abspath(r"..\data_cleaning\hall_calls_clean.csv")
    series = build_5min_series(csv_path)
    mu_w, phi_w = estimate_baseline_and_phi(series)
    print('Estimated phi_w:', phi_w)
    # show sample mu for weekday (first 12 intervals = midnight to 1am)
    print('Weekday mu (first 12):', mu_w[1][:12])
    print('Weekend mu (first 12):', mu_w[0][:12])
    # sample prediction for the last observed interval
    last_ts = series.index[-1]
    last_count = int(series['count'].iloc[-1])
    next_ts, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
    print(f'Sample next interval {next_ts} predicted count: {yhat:.3f} (last_count={last_count})')
