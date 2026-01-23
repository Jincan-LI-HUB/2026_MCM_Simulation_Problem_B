#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="task2_modes_timeline.csv",
                    help="CSV with Time + mode label (from notebook).")
    ap.add_argument("--out_fig", default="outputs/fig")
    ap.add_argument("--sample_days", type=int, default=1,
                    help="Plot first N days of the test window (default 1).")
    args = ap.parse_args()

    ensure_dir(args.out_fig)

    df = pd.read_csv(args.timeline)
    # infer time + mode columns
    tcol = None
    for c in ["Time", "time", "timestamp", "Datetime", "datetime"]:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise ValueError("Cannot find time column in timeline CSV.")
    df[tcol] = pd.to_datetime(df[tcol])

    mcol = None
    for c in ["mode_label", "mode", "label", "Mode", "cluster_label", "ClusterLabel"]:
        if c in df.columns:
            mcol = c
            break
    if mcol is None:
        # fallback: any non-time column
        candidates = [c for c in df.columns if c != tcol]
        if not candidates:
            raise ValueError("Cannot find mode column in timeline CSV.")
        mcol = candidates[0]

    df = df.sort_values(tcol).reset_index(drop=True)

    # Convert labels to categorical IDs for plotting
    labels = df[mcol].astype(str).unique().tolist()
    label2id = {lab: i for i, lab in enumerate(labels)}
    df["mode_id"] = df[mcol].astype(str).map(label2id)

    # 1) Timeline over first N days
    if args.sample_days > 0:
        t0 = df[tcol].min().normalize()
        t1 = t0 + pd.Timedelta(days=args.sample_days)
        sub = df[(df[tcol] >= t0) & (df[tcol] < t1)].copy()

        plt.figure()
        plt.step(sub[tcol], sub["mode_id"], where="post")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Time")
        plt.ylabel("Mode label")
        plt.tight_layout()
        out_pdf = os.path.join(args.out_fig, "task2_mode_timeline.pdf")
        plt.savefig(out_pdf)
        plt.close()
        print("Saved:", out_pdf)

    # 2) Heatmap: mode frequency by hour-of-day (rows=mode, cols=hour)
    df["hour"] = df[tcol].dt.hour
    heat = df.pivot_table(index=mcol, columns="hour", values="mode_id", aggfunc="count", fill_value=0)
    heat = heat.reindex(labels)  # keep label order

    plt.figure()
    plt.imshow(heat.values, aspect="auto")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(range(len(heat.columns)), heat.columns)
    plt.xlabel("Hour of day")
    plt.ylabel("Mode label")
    plt.tight_layout()
    out_pdf2 = os.path.join(args.out_fig, "task2_mode_heatmap.pdf")
    plt.savefig(out_pdf2)
    plt.close()
    print("Saved:", out_pdf2)


if __name__ == "__main__":
    main()
