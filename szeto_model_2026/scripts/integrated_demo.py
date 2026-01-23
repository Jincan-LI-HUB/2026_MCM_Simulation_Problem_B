#!/usr/bin/env python3
"""Integrated demo: Task 1 (forecast) + Task 2 (state classification).

Procedure:
- Build per-5-min features (counts, direction shares, etc.)
- Compute thresholds and classify every interval into a state
- Split intervals into train (first 80%) and test (last 20%)
- Estimate mu_w and phi_w on train (Task 1)
- One-step predict on test using previous observed interval
- Record y_true, y_hat (AR1), baseline, state, ts
- Compute overall and per-state MAE/RMSE and save results to CSV
"""
import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.task1_model import estimate_baseline_and_phi, predict_next, build_5min_series
from src.task2_classifier import build_interval_features, compute_thresholds, classify_all

OUT_CSV = os.path.join(repo_root, 'scripts', 'integrated_demo_results.csv')


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def main():
    csv_path = os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')

    print('Building interval features...')
    features = build_interval_features(csv_path)
    # ensure we have count,w,tau columns for task1 estimator
    series = features[['count', 'w', 'tau']].copy()

    print('Computing thresholds and classifying states...')
    thr = compute_thresholds(features)
    states = classify_all(features, thr)

    n = len(series)
    split = int(n * 0.8)
    print(f'Total intervals: {n}, train: {split}, test: {n - split}')

    train = series.iloc[:split]
    test_idx = series.index[split:]

    # Estimate parameters on train
    mu_w, phi_w = estimate_baseline_and_phi(train)
    print('Estimated phi_w (train):', phi_w)

    # Backtest one-step ahead on test portion
    full_counts = series['count'].values
    idx = series.index

    y_true = []
    y_model = []
    y_baseline = []
    ts_list = []
    state_list = []

    for pos in range(split, n):
        prev_pos = pos - 1
        last_ts = idx[prev_pos]
        last_count = int(full_counts[prev_pos])
        next_ts = idx[pos]

        _, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
        tau_tp1 = ((next_ts.hour * 60 + next_ts.minute) // 5) % (24 * 12)
        w_tp1 = 1 if next_ts.weekday() < 5 else 0
        baseline = mu_w[w_tp1][tau_tp1]

        y_true.append(float(full_counts[pos]))
        y_model.append(float(yhat))
        y_baseline.append(float(baseline))
        ts_list.append(next_ts)
        state_list.append(states.iloc[pos])

    df = pd.DataFrame({
        'ts': ts_list,
        'y_true': y_true,
        'y_model': y_model,
        'y_baseline': y_baseline,
        'state': state_list,
    })
    df['hour'] = df['ts'].dt.hour

    # Overall metrics
    overall = {
        'model_mae': mae(df['y_true'].values, df['y_model'].values),
        'model_rmse': rmse(df['y_true'].values, df['y_model'].values),
        'baseline_mae': mae(df['y_true'].values, df['y_baseline'].values),
        'baseline_rmse': rmse(df['y_true'].values, df['y_baseline'].values),
    }

    print('\nOverall one-step results:')
    print(f"Model MAE: {overall['model_mae']:.4f}, RMSE: {overall['model_rmse']:.4f}")
    print(f"Baseline MAE: {overall['baseline_mae']:.4f}, RMSE: {overall['baseline_rmse']:.4f}")

    # Per-state metrics
    grouped = df.groupby('state').apply(lambda g: pd.Series({
        'count': len(g),
        'model_mae': mae(g['y_true'].values, g['y_model'].values),
        'baseline_mae': mae(g['y_true'].values, g['y_baseline'].values),
        'model_rmse': rmse(g['y_true'].values, g['y_model'].values),
        'baseline_rmse': rmse(g['y_true'].values, g['y_baseline'].values),
    }))

    print('\nPer-state metrics:')
    print(grouped.sort_values('count', ascending=False))

    df.to_csv(OUT_CSV, index=False)
    print(f'Wrote integrated results to: {OUT_CSV}')

    print('\nSample rows:')
    print(df.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
