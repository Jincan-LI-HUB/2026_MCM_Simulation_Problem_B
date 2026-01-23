#!/usr/bin/env python3
"""Backtest Task 1 one-step forecasts.
Train on the first 80% of intervals, test on the last 20%.
Outputs overall MAE/RMSE for baseline-only and AR(1)-adjusted forecasts and saves per-interval CSV.
"""
import os
import sys
import numpy as np
import pandas as pd

# ensure repo root on path so we can import src.task1_model
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.task1_model import build_5min_series, estimate_baseline_and_phi, predict_next

CSV_PATH = os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')
OUT_CSV = os.path.join(repo_root, 'scripts', 'task1_backtest_results.csv')


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def main():
    series = build_5min_series(CSV_PATH)
    n = len(series)
    if n < 10:
        raise SystemExit('Not enough data')

    split = int(n * 0.8)
    train = series.iloc[:split]
    test = series.iloc[split:]

    print(f'Total intervals: {n}, train: {len(train)}, test: {len(test)}')

    mu_w, phi_w = estimate_baseline_and_phi(train)
    print('Estimated phi_w (from train):', phi_w)

    # Prepare arrays for backtest. We'll predict each test interval using previous observed interval.
    # We need the previous interval for the first test point; that will be train.iloc[-1]
    # Build positions: indexes in the full series
    full_counts = series['count'].values
    idx = series.index

    test_start_pos = split  # position of first test interval in full series
    y_true = []
    y_model = []
    y_baseline = []
    timestamps = []

    for pos in range(test_start_pos, n):
        prev_pos = pos - 1
        last_ts = idx[prev_pos]
        last_count = int(full_counts[prev_pos])
        next_ts = idx[pos]
        # model prediction using train mu/phi
        _, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
        # baseline-only prediction is mu_{w_{t+1}}(tau_{t+1})
        tau_tp1 = ((next_ts.hour * 60 + next_ts.minute) // 5) % (24 * 12)
        w_tp1 = 1 if next_ts.weekday() < 5 else 0
        baseline = mu_w[w_tp1][tau_tp1]

        y_true.append(float(full_counts[pos]))
        y_model.append(float(yhat))
        y_baseline.append(float(baseline))
        timestamps.append(next_ts)

    y_true = np.array(y_true)
    y_model = np.array(y_model)
    y_baseline = np.array(y_baseline)

    # Metrics
    metrics = {
        'model_mae': mae(y_true, y_model),
        'model_rmse': rmse(y_true, y_model),
        'baseline_mae': mae(y_true, y_baseline),
        'baseline_rmse': rmse(y_true, y_baseline),
    }

    print('\nBacktest results (one-step ahead):')
    print(f"Model MAE: {metrics['model_mae']:.4f}, RMSE: {metrics['model_rmse']:.4f}")
    print(f"Baseline MAE: {metrics['baseline_mae']:.4f}, RMSE: {metrics['baseline_rmse']:.4f}")

    # Hourly MAE comparison
    df_res = pd.DataFrame({'ts': timestamps, 'y_true': y_true, 'y_model': y_model, 'y_baseline': y_baseline})
    df_res['hour'] = df_res['ts'].dt.hour
    hourly = df_res.groupby('hour').apply(lambda g: pd.Series({
        'model_mae': mae(g['y_true'].values, g['y_model'].values),
        'baseline_mae': mae(g['y_true'].values, g['y_baseline'].values),
        'count': len(g)
    }))

    print('\nHourly MAE (sample):')
    print(hourly.head(12))

    # Save results
    df_res.to_csv(OUT_CSV, index=False)
    print(f'Wrote per-interval results to: {OUT_CSV}')

    # Print first 10 sample rows
    print('\nSample predictions (first 10 of test):')
    print(df_res.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
