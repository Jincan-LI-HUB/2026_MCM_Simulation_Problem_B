#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def infer_time_col(df: pd.DataFrame):
    for c in ["Time", "time", "timestamp", "Datetime", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find time column. Expected one of: Time/time/timestamp/Datetime/datetime")


def infer_floor_col(df: pd.DataFrame):
    for c in ["Floor", "floor", "FromFloor", "from_floor", "OriginFloor", "origin_floor"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find floor column. Expected one of: Floor/floor/FromFloor/from_floor/OriginFloor/origin_floor")


def build_5min_flow_series(hall_calls_path: str, freq="5min") -> pd.DataFrame:
    df = pd.read_csv(hall_calls_path)
    tcol = infer_time_col(df)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol)

    # Count hall calls per slice as demand proxy
    s = df.set_index(tcol).resample(freq).size().rename("y").reset_index()
    s = s.rename(columns={tcol: "Time"})
    return s


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["Time"])
    minute_of_day = t.dt.hour * 60 + t.dt.minute
    # daily cyclical
    df["sin_day"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["cos_day"] = np.cos(2 * np.pi * minute_of_day / 1440.0)

    # weekly cyclical (minute of week)
    minute_of_week = (t.dt.dayofweek * 1440) + minute_of_day
    df["sin_week"] = np.sin(2 * np.pi * minute_of_week / (7 * 1440.0))
    df["cos_week"] = np.cos(2 * np.pi * minute_of_week / (7 * 1440.0))

    df["hour"] = t.dt.hour
    df["minute"] = t.dt.minute
    df["is_weekend"] = (t.dt.dayofweek >= 5).astype(int)
    return df


def add_lag_and_rolling(df: pd.DataFrame, lags=(1, 2, 3, 6, 12), roll_windows=(3, 6, 12)) -> pd.DataFrame:
    for l in lags:
        df[f"lag_{l}"] = df["y"].shift(l)

    for w in roll_windows:
        df[f"roll_mean_{w}"] = df["y"].shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = df["y"].shift(1).rolling(w).std(ddof=0)

    # optional: 1-hour trend (12 * 5min = 60min)
    df["trend_1h"] = df["y"].shift(1) - df["y"].shift(12)
    return df


def write_booktabs_table(path_tex: str, caption: str, label: str, headers, rows):
    # Minimal escaping for LaTeX
    def esc(x):
        if isinstance(x, str):
            return x.replace("_", r"\_")
        return str(x)

    with open(path_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{{esc(caption)}}}\n")
        f.write(f"\\label{{{esc(label)}}}\n")
        f.write("\\begin{tabular}{lrr}\n\\toprule\n")
        f.write(" & ".join(map(esc, headers)) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(esc(v) for v in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="data/clean/hall_calls_clean.csv", help="Path to hall_calls_clean.csv")
    ap.add_argument("--freq", default="5min", help="Resample frequency, default 5min")
    ap.add_argument("--train_ratio", type=float, default=0.85, help="Chronological train split ratio")
    ap.add_argument("--out_fig", default="outputs/fig", help="Output figure folder")
    ap.add_argument("--out_tab", default="outputs/tab", help="Output table folder")
    ap.add_argument("--out_data", default="outputs/data", help="Output data folder")
    args = ap.parse_args()

    ensure_dir(args.out_fig)
    ensure_dir(args.out_tab)
    ensure_dir(args.out_data)

    df = build_5min_flow_series(args.hall, freq=args.freq)
    df = add_time_features(df)
    df = add_lag_and_rolling(df)

    df_feat = df.dropna().reset_index(drop=True)

    feature_cols = [
        "sin_day", "cos_day", "sin_week", "cos_week", "is_weekend",
        "hour", "minute",
    ]
    # add engineered columns
    feature_cols += [c for c in df_feat.columns if c.startswith("lag_") or c.startswith("roll_") or c == "trend_1h"]

    n = len(df_feat)
    n_train = int(math.floor(n * args.train_ratio))
    train = df_feat.iloc[:n_train].copy()
    test = df_feat.iloc[n_train:].copy()

    X_train, y_train = train[feature_cols].values, train["y"].values
    X_test, y_test = test[feature_cols].values, test["y"].values

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=400,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Save series for later use
    out_csv = os.path.join(args.out_data, "task1_pred_series.csv")
    pd.DataFrame({
        "Time": test["Time"].values,
        "y_true": y_test,
        "y_pred": y_pred
    }).to_csv(out_csv, index=False)

    # Save table (LaTeX)
    out_tex = os.path.join(args.out_tab, "task1_metrics.tex")
    write_booktabs_table(
        out_tex,
        caption="Task 1 forecasting performance (5-minute demand).",
        label="tab:task1-metrics",
        headers=["Metric", "Value", ""],
        rows=[
            ["MAE", f"{mae:.3f}", ""],
            ["RMSE", f"{rmse:.3f}", ""],
        ]
    )

    # Plot (PDF)
    # plot a representative window (first 288 points = 1 day for 5min slices), if available
    T = test["Time"].values
    m = min(len(T), 288)
    plt.figure()
    plt.plot(test["Time"].iloc[:m], y_test[:m], label="Observed")
    plt.plot(test["Time"].iloc[:m], y_pred[:m], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Hall calls per 5-min slice")
    plt.legend()
    plt.tight_layout()
    out_pdf = os.path.join(args.out_fig, "task1_pred_vs_true.pdf")
    plt.savefig(out_pdf)
    plt.close()

    print("Task1 materials generated:")
    print(" -", out_csv)
    print(" -", out_tex)
    print(" -", out_pdf)
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}")


if __name__ == "__main__":
    main()
