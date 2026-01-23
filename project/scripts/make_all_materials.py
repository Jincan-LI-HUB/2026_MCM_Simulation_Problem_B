#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_booktabs_table(path_tex: Path, caption: str, label: str, headers, rows, colfmt=None):
    def esc(x):
        if isinstance(x, str):
            return x.replace("_", r"\_")
        return str(x)

    if colfmt is None:
        colfmt = "l" + "r" * (len(headers) - 1)

    with open(path_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\caption{{{esc(caption)}}}\n")
        f.write(f"\\label{{{esc(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{colfmt}}}\n\\toprule\n")
        f.write(" & ".join(map(esc, headers)) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(esc(v) for v in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main():
    ROOT = Path.cwd()
    OUT_DATA = ROOT / "outputs" / "data"
    OUT_FIG = ROOT / "outputs" / "fig"
    OUT_TAB = ROOT / "outputs" / "tab"

    ensure_dir(OUT_FIG)
    ensure_dir(OUT_TAB)

    # -------------------------
    # Task1: prediction curve + metrics
    # -------------------------
    p1 = OUT_DATA / "task1_pred_series.csv"
    df1 = pd.read_csv(p1)
    df1["Time"] = pd.to_datetime(df1["Time"])
    df1 = df1.sort_values("Time")

    mae = mean_absolute_error(df1["y_true"], df1["y_pred"])
    mse = mean_squared_error(df1["y_true"], df1["y_pred"])
    rmse = float(np.sqrt(mse))

    # table
    write_booktabs_table(
        OUT_TAB / "task1_metrics.tex",
        caption="Task 1 forecasting performance (5-minute demand).",
        label="tab:task1-metrics",
        headers=["Metric", "Value"],
        rows=[["MAE", f"{mae:.3f}"], ["RMSE", f"{rmse:.3f}"]],
        colfmt="lr"
    )

    # plot (first day-ish window if available)
    m = min(len(df1), 288)  # 288 slices = 1 day at 5min
    plt.figure()
    plt.plot(df1["Time"].iloc[:m], df1["y_true"].iloc[:m], label="Observed")
    plt.plot(df1["Time"].iloc[:m], df1["y_pred"].iloc[:m], label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Hall calls per 5-min slice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG / "task1_pred_vs_true.pdf")
    plt.close()

   # -------------------------
# Regime transition figure: demand series + stable mode-boundary markers
# -------------------------
    p_tl = OUT_DATA / "task2_modes_timeline.csv"
    if p_tl.exists():
    # 1) load mode timeline
        tl = pd.read_csv(p_tl)
        tl["Time"] = pd.to_datetime(tl["Time"])
        tl = tl.sort_values("Time")

    # pick mode column
        mode_col = "mode_label" if "mode_label" in tl.columns else ("cluster" if "cluster" in tl.columns else None)
        if mode_col is None:
            print(f"[WARN] Cannot find mode column in {p_tl}. Columns={list(tl.columns)}")
        else:
            tl[mode_col] = tl[mode_col].astype(str)

    # 2) define plotting window from Task1 series
    # df1 must already exist in your script (loaded from task1_pred_series.csv)
        tmin = df1["Time"].iloc[0]
        tmax = df1["Time"].iloc[m-1]

    # 3) plot observed vs predicted demand
        plt.figure()
        plt.plot(df1["Time"].iloc[:m], df1["y_true"].iloc[:m], label="Observed demand")
        plt.plot(df1["Time"].iloc[:m], df1["y_pred"].iloc[:m], label="Predicted demand")

    # 4) draw stable regime boundaries (avoid dense "picket fence")
        if mode_col is not None:
            seg_id = (tl[mode_col] != tl[mode_col].shift(1)).cumsum()
            segs = (tl.assign(seg_id=seg_id)
                      .groupby("seg_id")
                      .agg(start=("Time", "min"),
                           end=("Time", "max"),
                           label=(mode_col, "first"),
                           n=("Time", "size"))
                      .reset_index())

        # keep segments lasting at least 3 slices (>= 15 minutes)
            segs = segs[segs["n"] >= 3].sort_values("start")

            boundaries = segs["start"].iloc[1:]  # start of each segment except first
            for t in boundaries:
                if tmin <= t <= tmax:
                    plt.axvline(t, linewidth=1.4, alpha=0.5)

        plt.xlabel("Time")
        plt.ylabel("Hall calls per 5-min slice")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_FIG / "regime_transition.pdf")
        plt.close()
    else:
        print(f"[WARN] Missing {p_tl}; skip regime_transition.pdf")




    # -------------------------
    # Task2: cluster profiles + mode heatmap
    # -------------------------
    p_feat = OUT_DATA / "task3_mode_features_5min.csv"
    df_feat = pd.read_csv(p_feat)
    df_feat["Time"] = pd.to_datetime(df_feat["Time"])

    # choose cluster column
    if "cluster" not in df_feat.columns:
        raise ValueError("task3_mode_features_5min.csv must contain column 'cluster'.")

    # choose columns to summarize (compact)
    preferred = [
        "hall_calls_w", "hall_calls", "activity_w", "activity",
        "up_ratio_w", "up_ratio",
        "entropy_w", "entropy",
        "car_inter_ratio_w", "car_inter_ratio", "inter_ratio_w", "inter_ratio",
        "stops_w", "stops",
        "departures_w", "departures",
        "net_load_w", "net_load",
        "maint_ratio_w", "maint_ratio",
    ]
    feat_cols = [c for c in preferred if c in df_feat.columns]
    if len(feat_cols) < 3:
        # fallback to numeric columns
        num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in num_cols if c != "cluster"][:6]

    grp = df_feat.dropna(subset=["cluster"]).groupby("cluster")
    prof = grp[feat_cols].mean()
    cnt = grp.size().rename("n").to_frame()
    prof = prof.join(cnt).reset_index().sort_values("cluster")

    # write table (limit to 6 feature cols)
    show_cols = feat_cols[:6]
    headers = ["Cluster", "n"] + show_cols
    rows = []
    for _, r in prof.iterrows():
        row = [int(r["cluster"]), int(r["n"])]
        for c in show_cols:
            row.append(f"{float(r[c]):.3f}")
        rows.append(row)

    write_booktabs_table(
        OUT_TAB / "task2_cluster_profile.tex",
        caption="Cluster profiles (mean feature values) used for interpretable traffic-mode naming.",
        label="tab:task2-cluster-profile",
        headers=headers,
        rows=rows,
        colfmt="l" + "r" * (len(headers) - 1)
    )

    # mode heatmap by hour-of-day
    p_tl = OUT_DATA / "task2_modes_timeline.csv"
    df_tl = pd.read_csv(p_tl)
    df_tl["Time"] = pd.to_datetime(df_tl["Time"])
    # use mode_label if exists else cluster
    if "mode_label" in df_tl.columns:
        key = "mode_label"
        df_tl[key] = df_tl[key].astype(str)
    else:
        key = "cluster"
        df_tl[key] = df_tl[key].astype(str)

    df_tl["hour"] = df_tl["Time"].dt.hour
    df_tl = df_tl.copy()
    df_tl["count"] = 1
    heat = df_tl.pivot_table(index=key, columns="hour", values="count",
                         aggfunc="sum", fill_value=0)

    plt.figure()
    plt.imshow(heat.values, aspect="auto")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.xlabel("Hour of day")
    plt.ylabel("Mode label")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "task2_mode_heatmap.pdf")
    plt.close()

    # -------------------------
    # Task3: strategy comparison table
    # -------------------------
    p_sim = OUT_DATA / "task3_sim_results.csv"
    if p_sim.exists():
        sim = pd.read_csv(p_sim)
        # expect columns: strategy, AWT_sec, LongWait_pct (names may vary)
        rename = {}
        if "AWT" in sim.columns and "AWT_sec" not in sim.columns:
            rename["AWT"] = "AWT_sec"
        if "LongWait%" in sim.columns and "LongWait_pct" not in sim.columns:
            rename["LongWait%"] = "LongWait_pct"
        sim = sim.rename(columns=rename)

        required = {"strategy", "AWT_sec", "LongWait_pct"}
        if not required.issubset(sim.columns):
            raise ValueError(f"task3_sim_results.csv must contain columns {required}, got {sim.columns.tolist()}")

        rows = []
        for _, r in sim.iterrows():
            rows.append([str(r["strategy"]), f"{float(r['AWT_sec']):.2f}", f"{float(r['LongWait_pct']):.2f}"])

        write_booktabs_table(
            OUT_TAB / "task3_strategy_comparison.tex",
            caption="Simulation comparison of parking strategies.",
            label="tab:task3-strategy",
            headers=["Strategy", "AWT (s)", "Long wait (\\%)"],
            rows=rows,
            colfmt="lrr"
        )
    else:
        # create placeholder table so LaTeX still compiles
        write_booktabs_table(
            OUT_TAB / "task3_strategy_comparison.tex",
            caption="Simulation comparison of parking strategies (placeholder; fill after exporting task3_sim_results.csv).",
            label="tab:task3-strategy",
            headers=["Strategy", "AWT (s)", "Long wait (\\%)"],
            rows=[["last_stop", "--", "--"], ["lobby", "--", "--"], ["dynamic", "--", "--"]],
            colfmt="lrr"
        )

    print("All materials generated under outputs/fig and outputs/tab.")


if __name__ == "__main__":
    main()

