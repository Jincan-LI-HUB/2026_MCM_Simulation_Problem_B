"""Demo runner for Task 3: runs a short integrated demo over a slice of historical data.

It will:
- load 5-min series from data_cleaning/hall_calls_clean.csv (Task1)
- estimate mu_w and phi_w
- build interval features and classify states (Task2)
- for a short window (default last 12 intervals ~1 hour), compute y_hat per interval and call parking_policy
- print a small table of timestamp,state,y_hat,desired_alloc,moves
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / 'src'))

from task1_model import build_5min_series, estimate_baseline_and_phi, predict_next
from task2_classifier import build_interval_features, compute_thresholds, classify_all
from task3_parking import parking_policy


def run_demo(csv_path: Optional[str] = None, n_elevators: int = 4, window: int = 12) -> None:
    if csv_path is None:
        csv_path = os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')
    print('Loading series...')
    series = build_5min_series(csv_path)
    print('Estimating baseline and phi...')
    mu_w, phi_w = estimate_baseline_and_phi(series)
    print('Building interval features...')
    features = build_interval_features(csv_path)
    thr = compute_thresholds(features)
    states = classify_all(features, thr)

    # choose window: last `window` intervals where we have observations
    end_idx = series.index[-1]
    start_idx = series.index[-1 - (window - 1)] if len(series) >= window else series.index[0]
    idxs = series.loc[start_idx:end_idx].index

    # current allocation: simple round-robin
    current_alloc = [ 'Lobby', 'Mid', 'Upper', 'Lobby' ][:n_elevators]

    print('\nDemo results (timestamp, state, y_hat -> desired_alloc, moves):')
    print('-' * 80)
    for ts in idxs:
        last_ts = ts
        last_count = int(series.loc[ts, 'count'])
        next_ts, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
        state = states.loc[ts]
        plan = parking_policy(state, yhat, n_elevators=n_elevators, current_alloc=current_alloc)
        print(f"{next_ts} | state={state[:15]:15} | y_hat={yhat:5.2f} | desired={plan['desired_alloc']} | moves={plan['moves']}")
    print('-' * 80)


if __name__ == '__main__':
    run_demo()