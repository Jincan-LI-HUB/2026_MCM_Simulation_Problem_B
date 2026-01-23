#!/usr/bin/env python3
"""
make_report_tables_v2.py

Drop-in replacement for make_report_tables.py, with two upgrades:
1) Long-wait threshold can be:
   - absolute seconds (default): --long_wait 60 --long_wait_mode absolute
   - relative to lobby baseline: --long_wait_mode p95_lobby (uses lobby P95 within the evaluated subset)
2) Task3 LaTeX overall table still compact; CSV outputs include EmptyTravel_mean if available.

Outputs (same names as before, so your LaTeX \input keeps working):
- outputs/report_table_task1.csv
- outputs/report_table_task1_by_state.csv
- outputs/report_table_task3_overall.csv
- outputs/report_table_task3_windows.csv
- outputs/latex_table_task1.tex
- outputs/latex_table_task3.tex
"""
from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd
from datetime import time

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pctl(x, q):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# Task 1 tables
# ---------------------------
def build_task1_tables(in_csv: str, out_dir: str):
    df = pd.read_csv(in_csv)
    ts_col = 'ts' if 'ts' in df.columns else ('interval_start' if 'interval_start' in df.columns else None)
    if ts_col is None:
        raise ValueError("Task1 CSV must include a timestamp column named 'ts' or 'interval_start'.")
    df[ts_col] = pd.to_datetime(df[ts_col])
    for c in ['y_true', 'y_model', 'y_baseline']:
        if c not in df.columns:
            raise ValueError(f"Task1 CSV missing required column: {c}")

    overall = pd.DataFrame([{
        'Model': 'Baseline+AR(1) (Route B)',
        'MAE': mae(df['y_true'], df['y_model']),
        'RMSE': rmse(df['y_true'], df['y_model']),
        'N': len(df)
    },{
        'Model': 'Time-of-day baseline (Route B)',
        'MAE': mae(df['y_true'], df['y_baseline']),
        'RMSE': rmse(df['y_true'], df['y_baseline']),
        'N': len(df)
    }])

    by_state = None
    if 'state' in df.columns:
        # explicit error columns to avoid pandas groupby.apply deprecations
        tmp = df.copy()
        tmp['abs_err_model'] = (tmp['y_true'] - tmp['y_model']).abs()
        tmp['abs_err_base']  = (tmp['y_true'] - tmp['y_baseline']).abs()
        tmp['sq_err_model']  = (tmp['y_true'] - tmp['y_model'])**2
        tmp['sq_err_base']   = (tmp['y_true'] - tmp['y_baseline'])**2

        by_state = (tmp.groupby('state', dropna=False)
                      .agg(N=('y_true','size'),
                           AR1_MAE=('abs_err_model','mean'),
                           BASE_MAE=('abs_err_base','mean'),
                           AR1_RMSE=('sq_err_model','mean'),
                           BASE_RMSE=('sq_err_base','mean'))
                      .reset_index())
        by_state['AR1_RMSE'] = np.sqrt(by_state['AR1_RMSE'])
        by_state['BASE_RMSE'] = np.sqrt(by_state['BASE_RMSE'])
        by_state = by_state.sort_values('N', ascending=False)

    overall.to_csv(os.path.join(out_dir, 'report_table_task1.csv'), index=False)
    if by_state is not None:
        by_state.to_csv(os.path.join(out_dir, 'report_table_task1_by_state.csv'), index=False)

    latex_lines = []
    latex_lines.append(r"\begin{table}[htbp]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Task 1 one-step forecasting benchmark (Route B).}")
    latex_lines.append(r"\label{tab:task1-benchmark-routeb}")
    latex_lines.append(r"\begin{tabular}{lrrr}")
    latex_lines.append(r"\hline")
    latex_lines.append(r"Model & MAE & RMSE & $N$ \\")
    latex_lines.append(r"\hline")
    for _, r in overall.iterrows():
        latex_lines.append(f"{r['Model']} & {r['MAE']:.3f} & {r['RMSE']:.3f} & {int(r['N'])} \\\\")
    latex_lines.append(r"\hline")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    with open(os.path.join(out_dir, 'latex_table_task1.tex'), 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))

# ---------------------------
# Task 3 windows tagging
# ---------------------------
def tag_windows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['call_time'] = pd.to_datetime(df['call_time'])
    df['weekday'] = df['call_time'].dt.weekday  # Mon=0
    df['tod'] = df['call_time'].dt.time

    def in_peak(t):
        return (time(8,0) <= t < time(10,0)) or (time(17,0) <= t < time(19,0))

    df['peak_window'] = (df['weekday'] <= 4) & df['tod'].map(in_peak)

    df['bin5'] = df['call_time'].dt.floor('5min')
    bin_counts = df.groupby('bin5').size().rename('bin_calls')
    thr = bin_counts.quantile(0.90)
    df = df.merge(bin_counts, left_on='bin5', right_index=True, how='left')
    df['high_load_window'] = df['bin_calls'] >= thr

    label_col = 'mode' if 'mode' in df.columns else ('state' if 'state' in df.columns else None)
    df['transition_window'] = False
    if label_col is not None:
        lab = (df[['bin5', label_col]]
               .drop_duplicates('bin5')
               .sort_values('bin5')
               .set_index('bin5')[label_col])
        changed = lab != lab.shift(1)
        change_bins = lab.index[changed.fillna(False)]
        bin_set = set()
        for b in change_bins:
            for k in range(-3, 4):
                bin_set.add(b + pd.Timedelta(minutes=5*k))
        df['transition_window'] = df['bin5'].isin(bin_set)

    return df

def compute_long_wait_threshold(df: pd.DataFrame, mode: str, absolute_value: float) -> float:
    mode = mode.lower().strip()
    if mode == "absolute":
        return float(absolute_value)
    if mode == "p95_lobby":
        lobby = df[df['strategy'].astype(str).str.lower().eq('lobby')]
        if len(lobby) == 0:
            # fallback: use overall p95
            return float(pctl(df['wait_time'], 95))
        return float(pctl(lobby['wait_time'], 95))
    raise ValueError("Unsupported --long_wait_mode. Use 'absolute' or 'p95_lobby'.")

def summarize_task3(df: pd.DataFrame, long_wait_threshold: float) -> pd.DataFrame:
    df = df.copy()
    if 'is_long_wait' not in df.columns:
        df['is_long_wait'] = (df['wait_time'] >= long_wait_threshold).astype(int)

    rows = []
    for strat, g in df.groupby('strategy'):
        rows.append({
            'strategy': strat,
            'N': len(g),
            'AWT': float(np.mean(g['wait_time'])),
            'P90': pctl(g['wait_time'], 90),
            'P95': pctl(g['wait_time'], 95),
            'P99': pctl(g['wait_time'], 99),
            'LongWait%': float(100 * np.mean(g['is_long_wait'])),
            'EmptyTravel_mean': float(np.mean(g['empty_travel'])) if 'empty_travel' in g.columns else np.nan,
        })
    return pd.DataFrame(rows).sort_values('AWT')

def build_task3_tables(in_csv: str, out_dir: str, long_wait_abs: float, long_wait_mode: str):
    df = pd.read_csv(in_csv)
    for c in ['strategy', 'call_time', 'wait_time']:
        if c not in df.columns:
            raise ValueError(f"Task3 CSV missing required column: {c}")

    df = tag_windows(df)

    # overall threshold
    thr_overall = compute_long_wait_threshold(df, long_wait_mode, long_wait_abs)
    overall = summarize_task3(df, thr_overall)
    overall.to_csv(os.path.join(out_dir, 'report_table_task3_overall.csv'), index=False)

    window_defs = [
        ('Peak (weekday 08-10,17-19)', df['peak_window']),
        ('Transition (±15min around label changes)', df['transition_window']),
        ('High-load (top 10% 5-min bins)', df['high_load_window']),
    ]

    window_rows = []
    for wname, mask in window_defs:
        gdf = df.loc[mask]
        if len(gdf) == 0:
            continue
        thr_w = compute_long_wait_threshold(gdf, long_wait_mode, long_wait_abs)
        summ = summarize_task3(gdf, thr_w)
        summ.insert(0, 'window', wname)
        summ.insert(1, 'LongWaitThreshold', thr_w)
        window_rows.append(summ)

    windows = pd.concat(window_rows, ignore_index=True) if window_rows else pd.DataFrame()
    windows.to_csv(os.path.join(out_dir, 'report_table_task3_windows.csv'), index=False)

    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Task 3 strategy comparison (overall).}")
    latex.append(r"\label{tab:task3-strategy}")
    latex.append(r"\begin{tabular}{lrrrrr}")
    latex.append(r"\hline")
    latex.append(r"Strategy & AWT & P95 & P99 & Long wait (\%) & $N$ \\")
    latex.append(r"\hline")
    for _, r in overall.iterrows():
        lw = r['LongWait%']
        latex.append(f"{str(r['strategy']).replace('_', r'\\_')} & {r['AWT']:.2f} & {r['P95']:.2f} & {r['P99']:.2f} & {lw:.2f} & {int(r['N'])} \\\\")
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    with open(os.path.join(out_dir, 'latex_table_task3.tex'), 'w', encoding='utf-8') as f:
        f.write("\n".join(latex))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo_root', default='.', help='Path to repo root (contains scripts/, outputs/, data_cleaning/)')
    ap.add_argument('--task1_csv', default='scripts/integrated_demo_results.csv')
    ap.add_argument('--task3_csv', default='outputs/task3_call_metrics.csv')
    ap.add_argument('--out_dir', default='outputs')
    ap.add_argument('--long_wait', type=float, default=60.0, help='Absolute long-wait threshold (used if --long_wait_mode absolute).')
    ap.add_argument('--long_wait_mode', default='absolute', choices=['absolute', 'p95_lobby'],
                    help="How to define 'long wait': absolute threshold, or P95 of lobby baseline within the subset.")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    out_dir = os.path.join(repo, args.out_dir)
    ensure_dir(out_dir)

    task1_path = os.path.join(repo, args.task1_csv)
    task3_path = os.path.join(repo, args.task3_csv)

    if os.path.exists(task1_path):
        build_task1_tables(task1_path, out_dir)
        print(f"[OK] Task1 tables written to {out_dir}")
    else:
        print(f"[SKIP] Task1 CSV not found: {task1_path}")

    if os.path.exists(task3_path):
        build_task3_tables(task3_path, out_dir, args.long_wait, args.long_wait_mode)
        print(f"[OK] Task3 tables written to {out_dir}")
    else:
        print(f"[SKIP] Task3 CSV not found: {task3_path}")
        print("      Export call-level metrics as outputs/task3_call_metrics.csv")

if __name__ == '__main__':
    main()
