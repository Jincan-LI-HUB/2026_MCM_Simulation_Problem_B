"""Task 2: Real-time traffic state classification (rule-based) per the tentative report.

This module computes real-time features per 5-minute interval from `hall_calls_clean.csv` and
applies a transparent rule-based classifier that maps each interval to one of the states:
  ['Night Idle','Morning Up-Peak','Lunch Down-Peak','Afternoon Mixed','Evening Down-Peak','Weekend Low-Demand']

Design decisions / assumptions (brief):
- Thresholds for demand intensity (theta1, theta2, theta3) are estimated from training data using quantiles.
  Default quantiles: theta1=10th percentile (low activity), theta2=75th percentile (high activity), theta3=75th percentile.
- Direction dominance thresholds: alpha1=0.6 (strong upward), alpha2=0.4 (strong downward), alpha3=0.6 (used for lunch middle band).
- Beta1 (lobby concentration) defaults to 0.3 (30% of calls from floor 1 suggests lobby concentration).

You can override thresholds by calling compute_thresholds and passing custom values.
"""
from __future__ import annotations
import os
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum() if counts.sum() > 0 else np.array([])
    p = p[p > 0]
    return float((- (p * np.log2(p)).sum()) if p.size > 0 else 0.0)


def build_interval_features(csv_path: str) -> pd.DataFrame:
    """Load hall calls CSV and compute per-5-minute interval features.

    Returns DataFrame indexed by interval_start with columns:
      count, cup (up count), cdown (down count), r (proportion up), p1 (prop from floor 1), H (spatial entropy), w, tau
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    # detect time and direction and floor columns
    time_col = None
    for c in df.columns:
        if c.lower() in ('time', 'timestamp', 'datetime') or 'time' in c.lower():
            time_col = c
            break
    if time_col is None:
        raise ValueError('No time column found in CSV')
    dir_col = None
    for c in df.columns:
        if c.lower() in ('direction', 'dir'):
            dir_col = c
            break
    if dir_col is None:
        raise ValueError('No direction column found in CSV')
    floor_col = None
    for c in df.columns:
        if 'floor' in c.lower():
            floor_col = c
            break
    if floor_col is None:
        raise ValueError('No floor column found in CSV')

    df[time_col] = pd.to_datetime(df[time_col])
    df['interval_start'] = df[time_col].dt.floor('5T')

    # Compute counts
    grp = df.groupby('interval_start')
    total_counts = grp.size().rename('count')
    up_counts = grp.apply(lambda g: (g[dir_col].str.lower() == 'up').sum()).rename('cup')
    down_counts = grp.apply(lambda g: (g[dir_col].str.lower() == 'down').sum()).rename('cdown')

    # per-interval floor distributions for entropy and p1
    def floor_stats(g):
        floors = g[floor_col]
        vc = floors.value_counts()
        p1 = float((floors == 1).sum())
        H = _entropy(vc.values)
        return pd.Series({'p1': p1, 'H': H})

    floor_df = grp.apply(floor_stats)

    # Combine, reindex full timeline
    idx = pd.date_range(start=total_counts.index.min(), end=total_counts.index.max(), freq='5T')
    out = pd.DataFrame(index=idx)
    out.index.name = 'interval_start'
    out = out.join(total_counts.reindex(idx, fill_value=0))
    out = out.join(up_counts.reindex(idx, fill_value=0))
    out = out.join(down_counts.reindex(idx, fill_value=0))
    out = out.join(floor_df.reindex(idx, fill_value=0))

    out['r'] = out['cup'] / out['count'].replace(0, np.nan)
    out['r'] = out['r'].fillna(0.0)
    out['p1'] = out['p1'] / out['count'].replace(0, np.nan)
    out['p1'] = out['p1'].fillna(0.0)
    out['w'] = out.index.to_series().dt.weekday.apply(lambda d: 1 if d < 5 else 0).astype(int).values
    out['tau'] = ((out.index.hour * 60 + out.index.minute) // 5).astype(int)

    return out


def compute_thresholds(features: pd.DataFrame, quantiles: Dict[str, float] = None) -> Dict[str, Any]:
    """Compute thresholds from provided features DataFrame.

    Returns a dict with keys theta1, theta2, theta3, alpha1, alpha2, alpha3, beta1
    """
    if quantiles is None:
        quantiles = {'theta1': 0.10, 'theta2': 0.75, 'theta3': 0.75}

    theta1 = float(features['count'].quantile(quantiles.get('theta1', 0.1)))
    theta2 = float(features['count'].quantile(quantiles.get('theta2', 0.75)))
    theta3 = float(features['count'].quantile(quantiles.get('theta3', 0.75)))
    # direction thresholds (interpretable defaults)
    alpha1 = 0.6
    alpha2 = 0.4
    alpha3 = 0.6
    beta1 = 0.3

    return {
        'theta1': theta1,
        'theta2': theta2,
        'theta3': theta3,
        'alpha1': alpha1,
        'alpha2': alpha2,
        'alpha3': alpha3,
        'beta1': beta1,
    }


def classify_interval(row: pd.Series, thr: Dict[str, Any]) -> str:
    """Classify a single interval (row from features) into a traffic state."""
    C = float(row['count'])
    r = float(row['r'])
    p1 = float(row['p1'])
    w = int(row['w'])

    theta1 = thr['theta1']
    theta2 = thr['theta2']
    theta3 = thr['theta3']
    alpha1 = thr['alpha1']
    alpha2 = thr['alpha2']
    alpha3 = thr['alpha3']
    beta1 = thr['beta1']

    # Decision order follows a sensible interpretation of the report
    if C < theta1:
        return 'Night Idle'
    if w == 0:
        return 'Weekend Low-Demand'
    # Morning Up-Peak: high demand, strong upward and lobby concentration
    if (C >= theta2) and (r >= alpha1) and (p1 >= beta1):
        return 'Morning Up-Peak'
    # Evening Down-Peak: high demand and strong downward dominance
    if (C >= theta3) and (r <= alpha2):
        return 'Evening Down-Peak'
    # Lunch Down-Peak: high demand, moderate downward dominance
    if (C >= theta2) and (alpha2 < r < alpha3):
        return 'Lunch Down-Peak'
    # Otherwise
    return 'Afternoon Mixed'


def classify_all(features: pd.DataFrame, thr: Dict[str, Any] = None) -> pd.Series:
    """Classify every interval in features. If thr is None, compute from features.

    Returns a pandas Series of state labels indexed by interval_start.
    """
    if thr is None:
        thr = compute_thresholds(features)
    states = features.apply(lambda row: classify_interval(row, thr), axis=1)
    return states


if __name__ == '__main__':
    # quick demo when run directly
    import sys
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')
    features = build_interval_features(csv_path)
    thr = compute_thresholds(features)
    print('Computed thresholds:', thr)
    states = classify_all(features, thr)
    print('State counts:')
    print(states.value_counts())
    print('\nSample classifications:')
    print(states.head(20))
