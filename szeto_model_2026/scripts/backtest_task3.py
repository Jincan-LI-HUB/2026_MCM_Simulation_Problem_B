#!/usr/bin/env python3
"""Backtest Task 3: simulate parking policy over historical data and compute simple metrics.

Simulation assumptions (simple, interpretable):
- Discrete 5-minute intervals (matching Task 1/Task 2).
- Fleet of N elevators. Each elevator parked in a zone contributes a fixed service capacity
  (calls served per 5-minute interval) when demand is directed to that zone.
- Service capacity per elevator per interval is a tunable parameter (default 6 calls / 5min).
- AWT proxy (seconds): baseline_wait + 60 * max(0, (demand - capacity) / max(1, capacity)).
  baseline_wait defaults to 10s. This yields low wait when capacity >= demand and grows
  proportionally to the relative overload.
- Empty-travel proxy: repositioning time per move (seconds) times number of moves, divided by
  total elevator-interval time.

Outputs:
- Printed summary of average AWT proxy, max AWT, fraction of intervals exceeding tau (long-wait),
  average moves per interval, and empty-travel share.
- Per-interval CSV saved to `scripts/task3_backtest_results.csv`.

This is intentionally simple so results are interpretable; we can later upgrade to a full motion
simulator if you prefer.
"""
from __future__ import annotations
import os
import sys
from typing import Dict
import numpy as np
import pandas as pd

# ensure repo root on path so we can import src modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.task1_model import build_5min_series, estimate_baseline_and_phi, predict_next
from src.task2_classifier import build_interval_features, compute_thresholds, classify_all
from src.task3_parking import parking_policy

CSV_PATH = os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')
OUT_CSV = os.path.join(repo_root, 'scripts', 'task3_backtest_results.csv')

# Simulation parameters
N_ELEVATORS = 4
SERVICE_PER_ELEV_PER_INTERVAL = 6.0  # calls an elevator can serve in one 5-min interval
BASELINE_WAIT_S = 10.0
REPOSITION_TIME_PER_MOVE_S = 30.0
LONG_WAIT_THRESHOLD_S = 60.0  # tau
TRAIN_FRACTION = 0.8


def run_backtest(
    csv_path: str = CSV_PATH,
    n_elevators: int = N_ELEVATORS,
    service_per_elev: float = SERVICE_PER_ELEV_PER_INTERVAL,
    baseline_wait_s: float = BASELINE_WAIT_S,
    reposition_time_per_move_s: float = REPOSITION_TIME_PER_MOVE_S,
    long_wait_threshold_s: float = LONG_WAIT_THRESHOLD_S,
    train_fraction: float = TRAIN_FRACTION,
):
    series = build_5min_series(csv_path)
    features = build_interval_features(csv_path)

    n = len(series)
    if n < 10:
        raise SystemExit('Not enough data')

    split = int(n * train_fraction)
    train = series.iloc[:split]
    test = series.iloc[split:]

    print(f'Total intervals: {n}, train: {len(train)}, test: {len(test)}')

    # Train Task1 baseline and phi on train
    mu_w, phi_w = estimate_baseline_and_phi(train)

    # compute thresholds and states on full features (we use observed features for classification)
    thr = compute_thresholds(features)
    states = classify_all(features, thr)

    idx = series.index
    full_counts = series['count'].values

    records = []
    total_reposition_time = 0.0

    # default current allocation: round-robin
    current_alloc = ["Lobby", "Mid", "Upper", "Lobby"][:n_elevators]

    for pos in range(split, n):
        prev_pos = pos - 1
        last_ts = idx[prev_pos]
        last_count = int(full_counts[prev_pos])
        next_ts = idx[pos]

        # prediction from model trained on train
        _, yhat = predict_next(last_ts, last_count, mu_w, phi_w)

        # choose state using the classifier for the previous (observed) interval
        state = states.iloc[prev_pos]

        plan = parking_policy(state, yhat, n_elevators=n_elevators, current_alloc=current_alloc)
        desired_alloc = plan['desired_alloc']
        moves = plan['moves']

        # compute capacity: number of elevators allocated * service per elevator
        allocated_elevators = sum(desired_alloc.get(z, 0) for z in ('Lobby', 'Mid', 'Upper'))
        capacity = allocated_elevators * service_per_elev
        if capacity < 1e-6:
            capacity = 1e-6

        y_true = float(full_counts[pos])

        # Simple AWT proxy (seconds)
        awt = baseline_wait_s + 60.0 * max(0.0, (y_true - capacity) / capacity)

        moves_count = len(moves)
        reposition_time = moves_count * reposition_time_per_move_s
        total_reposition_time += reposition_time

        records.append({
            'observed_interval': last_ts,
            'pred_interval': next_ts,
            'state': state,
            'y_hat': float(yhat),
            'y_true': y_true,
            'allocated_elevators': int(allocated_elevators),
            'capacity': float(capacity),
            'awt_s': float(awt),
            'moves_count': int(moves_count),
            'reposition_time_s': float(reposition_time),
        })

        # assume after moves we now have current_alloc matching desired for next step
        # expand desired_alloc to a list for current_alloc for deterministic behavior
        new_alloc = []
        for z in ('Lobby', 'Mid', 'Upper'):
            new_alloc.extend([z] * desired_alloc.get(z, 0))
        # if length differs due to rounding, pad with Lobby
        while len(new_alloc) < n_elevators:
            new_alloc.append('Lobby')
        current_alloc = new_alloc[:n_elevators]

    df = pd.DataFrame(records)

    # Metrics
    avg_awt = df['awt_s'].mean()
    max_awt = df['awt_s'].max()
    long_wait_frac = (df['awt_s'] > long_wait_threshold_s).mean()
    avg_moves = df['moves_count'].mean()

    # Empty-travel fraction: total reposition time / total available elevator time
    interval_seconds = 5 * 60
    total_elevator_time = len(df) * n_elevators * interval_seconds
    empty_travel_fraction = total_reposition_time / total_elevator_time

    print('\nBacktest Task 3 summary:')
    print(f'Average AWT proxy (s): {avg_awt:.2f}')
    print(f'Max AWT proxy (s): {max_awt:.2f}')
    print(f'Fraction intervals with AWT > {long_wait_threshold_s}s: {long_wait_frac:.3f}')
    print(f'Average moves per interval: {avg_moves:.3f}')
    print(f'Empty-travel fraction (reposition time share): {empty_travel_fraction:.4f}')

    # Save per-interval results
    df.to_csv(OUT_CSV, index=False)
    print(f'Per-interval results written to: {OUT_CSV}')

    return {
        'avg_awt_s': avg_awt,
        'max_awt_s': max_awt,
        'long_wait_frac': long_wait_frac,
        'avg_moves': avg_moves,
        'empty_travel_fraction': empty_travel_fraction,
        'out_csv': OUT_CSV,
    }


if __name__ == '__main__':
    run_backtest()
