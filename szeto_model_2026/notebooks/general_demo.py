"""General demo script combining Task1, Task2, Task3 demos and a lightweight backtest.

This is a straight export of `notebooks/general_demo.ipynb` as a Python script so you can edit
and run it easily from an editor or terminal. It is intentionally linear and contains
sections that correspond to the original notebook cells. Run it from the repo root or
from the `notebooks/` directory; it will try to import `src` accordingly.

Note: This script will attempt to import project modules from `src/` and the backtest
script from `scripts/`. It does not run automatically in this environment.
"""
from __future__ import annotations

from pathlib import Path
import sys
import importlib.util

# Notebook-style configuration (edit these parameters)
repo_root = Path(__file__).resolve().parents[1]
csv_path = repo_root / 'data_cleaning' / 'hall_calls_clean.csv'
n_elevators = 4
service_per_elev = 6.0  # calls per elevator per 5-min interval
baseline_wait_s = 10.0
reposition_time_s = 30.0
window = 60  # number of recent intervals to inspect in quick demos

# ensure src is importable
sys.path.insert(0, str(repo_root / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from task1_model import build_5min_series, estimate_baseline_and_phi, predict_next
from task2_classifier import build_interval_features, compute_thresholds, classify_all
from task3_parking import parking_policy


def task1_demo(csv_path: Path, window: int = 60):
    """Run Task1 parts: build series, estimate baseline and phi, sample predictions."""
    series = build_5min_series(str(csv_path))
    print('Series length:', len(series))
    mu_w, phi_w = estimate_baseline_and_phi(series)
    print('Estimated phi_w:', phi_w)

    # plot weekday/weekend baseline for first 48 intervals (4 hours)
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        ax.plot(mu_w[1][:48], label='weekday mu (first 4h)')
        ax.plot(mu_w[0][:48], label='weekend mu (first 4h)')
        ax.set_xlabel('5-min interval index')
        ax.set_ylabel('mean calls')
        ax.legend()
        plt.show()
    except Exception:
        # plotting may not be available in headless runs; ignore
        pass

    idxs = series.index[-window:] if len(series) >= window else series.index
    preds = []
    for ts in idxs:
        last_ts = ts
        last_count = int(series.loc[ts, 'count'])
        next_ts, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
        preds.append({'observed': last_ts, 'pred_interval': next_ts, 'y_hat': yhat})
    df_preds = pd.DataFrame(preds)
    print('\nSample Task1 predictions:')
    print(df_preds.head())
    return series, mu_w, phi_w


def task2_demo(csv_path: Path):
    """Run Task2 parts: build interval features and classify states."""
    features = build_interval_features(str(csv_path))
    thr = compute_thresholds(features)
    states = classify_all(features, thr)
    print('\nComputed thresholds:', thr)
    print('\nState counts:')
    print(states.value_counts())
    return features, thr, states


def task3_integrated_demo(series, mu_w, phi_w, states, n_elevators: int = 4):
    """Integrated short demo: compute y_hat and parking plans for a short window."""
    # choose a short window near the end of the series
    idxs = series.index[-window:] if len(series) >= window else series.index
    current_alloc = ['Lobby', 'Mid', 'Upper', 'Lobby'][:n_elevators]
    rows = []
    for ts in idxs:
        last_ts = ts
        last_count = int(series.loc[ts, 'count'])
        next_ts, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
        state = states.loc[ts]
        plan = parking_policy(state, yhat, n_elevators=n_elevators, current_alloc=current_alloc)
        rows.append({
            'observed': ts,
            'pred_interval': next_ts,
            'state': state,
            'y_hat': yhat,
            'desired': plan['desired_alloc'],
            'moves': plan['moves'],
        })
        # update current allocation deterministically to desired
        new_alloc = []
        for z in ('Lobby', 'Mid', 'Upper'):
            new_alloc.extend([z] * plan['desired_alloc'].get(z, 0))
        while len(new_alloc) < n_elevators:
            new_alloc.append('Lobby')
        current_alloc = new_alloc[:n_elevators]

    df = pd.DataFrame(rows)
    print('\nTask3 integrated demo (sample rows):')
    print(df.head())
    return df


def run_backtest_from_script(csv_path: Path, **kwargs):
    """Import and call the backtest script `scripts/backtest_task3.py`'s run_backtest function.
    We import it as a module by file path to avoid packaging requirements.
    """
    backtest_path = repo_root / 'scripts' / 'backtest_task3.py'
    spec = importlib.util.spec_from_file_location('backtest_task3', str(backtest_path))
    backtest_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(backtest_mod)  # type: ignore
    res = backtest_mod.run_backtest(csv_path=str(csv_path), **kwargs)
    return res


def main():
    print('Running general_demo.py (script export of notebook).')
    print('CSV path:', csv_path)

    # Task1
    series, mu_w, phi_w = task1_demo(csv_path, window=window)

    # Task2
    features, thr, states = task2_demo(csv_path)

    # Task3 integrated demo
    df_demo = task3_integrated_demo(series, mu_w, phi_w, states, n_elevators=n_elevators)

    # Backtest (call the backtest module)
    print('\nRunning backtest... (this will write a per-interval CSV)')
    try:
        res = run_backtest_from_script(csv_path, n_elevators=n_elevators, service_per_elev=service_per_elev,
                                       baseline_wait_s=baseline_wait_s,
                                       reposition_time_per_move_s=reposition_time_s,
                                       train_fraction=0.8)
        print('\nBacktest returned:', res)
        # try to load and show a few rows of the produced CSV
        out_csv = res.get('out_csv')
        if out_csv:
            df_bt = pd.read_csv(out_csv, parse_dates=['observed_interval', 'pred_interval'])
            print('\nPer-interval backtest sample:')
            print(df_bt.head())
    except Exception as e:
        print('Backtest failed to run in this environment (expected if running headless):', e)


if __name__ == '__main__':
    main()
