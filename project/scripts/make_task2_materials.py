#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_booktabs_table(path_tex: str, caption: str, label: str, headers, rows, colfmt=None):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode_feat", default="task3_mode_features_5min.csv",
                    help="CSV with per-slice features + cluster label (from notebook).")
    ap.add_argument("--out_fig", default="outputs/fig")
    ap.add_argument("--out_tab", default="outputs/tab")
    args = ap.parse_args()

    ensure_dir(args.out_fig)
    ensure_dir(args.out_tab)

    df = pd.read_csv(args.mode_feat)
    # infer cluster column
    cluster_col = None
    for c in ["cluster", "mode_cluster", "kmeans_cluster", "Cluster", "mode_id"]:
        if c in df.columns:
            cluster_col = c
            break
    if cluster_col is None:
        raise ValueError("Cannot find cluster column in mode features CSV.")

    # pick interpretable feature columns (use what exists)
    preferred = [
        "hall_calls_w", "hall_calls",
        "up_ratio_w", "up_ratio",
        "entropy_w", "entropy",
        "car_inter_ratio_w", "car_inter_ratio", "inter_ratio_w", "inter_ratio",
        "stops_w", "stops",
        "departures_w", "departures",
        "net_load_w", "net_load",
        "maint_ratio_w", "maint_ratio",
    ]
    feat_cols = [c for c in preferred if c in df.columns]
    if len(feat_cols) < 3:
        # fallback: take numeric columns excluding time and cluster
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_cols = [c for c in numeric if c != cluster_col][:8]

    grp = df.groupby(cluster_col)
    prof = grp[feat_cols].mean().reset_index()
    cnt = grp.size().reset_index(name="n_slices")
    prof = prof.merge(cnt, on=cluster_col, how="left").sort_values(cluster_col)

    # --- write LaTeX table (compact) ---
    headers = ["Cluster", "n"] + feat_cols[:6]  # keep compact for MCM pages
    rows = []
    for _, r in prof.iterrows():
        row = [int(r[cluster_col]), int(r["n_slices"])]
        for c in feat_cols[:6]:
            row.append(f"{float(r[c]):.3f}")
        rows.append(row)

    out_tex = os.path.join(args.out_tab, "task2_cluster_profile.tex")
    write_booktabs_table(
        out_tex,
        caption="Cluster profiles (mean feature values) used for interpretable traffic-mode naming.",
        label="tab:task2-cluster-profile",
        headers=headers,
        rows=rows,
        colfmt="l" + "r" * (len(headers) - 1)
    )

    # --- optional: bar plot for 3 key features (saves space vs radar) ---
    key3 = feat_cols[:3]
    if len(key3) >= 3:
        x = np.arange(len(prof))
        plt.figure()
        for c in key3:
            plt.plot(x, prof[c].values, marker="o", label=c)
        plt.xticks(x, [str(int(v)) for v in prof[cluster_col].values])
        plt.xlabel("Cluster ID")
        plt.ylabel("Mean value")
        plt.legend()
        plt.tight_layout()
        out_pdf = os.path.join(args.out_fig, "task2_cluster_profile_plot.pdf")
        plt.savefig(out_pdf)
        plt.close()
    else:
        out_pdf = None

    print("Task2 materials generated:")
    print(" -", out_tex)
    if out_pdf:
        print(" -", out_pdf)
    print("Columns used:", headers)


if __name__ == "__main__":
    main()
