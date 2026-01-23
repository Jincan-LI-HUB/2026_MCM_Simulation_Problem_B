#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/task3_evaluator_parking.py

What this does
--------------
Generate call-level Task 3 evaluation logs:
    outputs/task3_call_metrics.csv

Why your tables were identical
------------------------------
If your evaluator computes wait_time only from "nearest car after serving calls" and never
applies different parking targets per strategy, then all strategies will produce identical
wait_time distributions. This script explicitly applies *different parking floors* every
5 minutes (parking decision epoch), so strategies diverge.

Modeling level (intentionally simple, competition-friendly)
-----------------------------------------------------------
- Time is discretized into 5-minute epochs.
- At the start of each epoch, each strategy sets k parking floors for idle cars:
    * lobby: all cars at floor 1
    * zone: cars distributed across (Lobby/Mid/Upper) using recent demand by zone
    * dynamic: k "weighted-quantile" medians from recent floor distribution (1D k-median surrogate)
    * last_stop: keep the previous epoch's parking floors
- Within the epoch, each call wait_time is approximated by:
        wait_time = min_j |floor - park_j| * V_FLOOR + DOOR_TIME
  (consistent distance-proxy logic; relative comparisons are the point)
- Empty-travel is approximated as total movement when changing parking floors between epochs.

Inputs
------
- data_cleaning/hall_calls_clean.csv  (must contain Time, Floor; Direction optional)

Outputs
-------
- outputs/task3_call_metrics.csv with columns:
    strategy, call_time, wait_time, state, empty_travel
"""

from __future__ import annotations
import os
import math
import argparse
import numpy as np
import pandas as pd

# -------------------------
# utilities
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def weighted_quantile_floors(floors: np.ndarray, k: int) -> list[int]:
    """
    Floors are integers, repeated by frequency (already expanded by counts).
    Returns k parking floors as quantiles of the empirical distribution.
    """
    if len(floors) == 0:
        return []
    floors_sorted = np.sort(floors.astype(int))
    # quantile positions centered in each k-segment
    qs = [(i + 0.5) / k for i in range(k)]
    idxs = [min(len(floors_sorted) - 1, max(0, int(round(q * (len(floors_sorted) - 1))))) for q in qs]
    return [int(floors_sorted[i]) for i in idxs]

def match_sorted_distance(prev: list[int], curr: list[int]) -> float:
    """
    Approximate minimum matching cost in 1D by sorting both lists and pairing.
    """
    if len(prev) == 0 or len(curr) == 0:
        return 0.0
    a = np.sort(np.array(prev, dtype=float))
    b = np.sort(np.array(curr, dtype=float))
    n = min(len(a), len(b))
    return float(np.sum(np.abs(a[:n] - b[:n])))

def compute_state(count: int, up_ratio: float, is_weekday: bool, hour: int, q25: float, q75: float) -> str:
    """
    Simple, transparent state logic (for transition-window tagging).
    You can replace this with your Route B classifier later.
    """
    if (count <= q25):
        return "Idle/Low"
    if is_weekday and (7 <= hour <= 10) and (up_ratio >= 0.60) and (count >= q75):
        return "Morning Up-Peak"
    if is_weekday and (16 <= hour <= 19) and (up_ratio <= 0.40) and (count >= q75):
        return "Evening Down-Peak"
    if is_weekday and (11 <= hour <= 14) and (up_ratio <= 0.45) and (count >= q75):
        return "Lunch Down-Peak"
    return "Mixed"

def zone_of_floor(f: int, lobby_max: int, mid_max: int) -> str:
    if f <= lobby_max:
        return "Lobby"
    if f <= mid_max:
        return "Mid"
    return "Upper"

# -------------------------
# parking policies
# -------------------------

def parking_lobby(k: int) -> list[int]:
    return [1] * k

def parking_last_stop(prev_parking: list[int], k: int) -> list[int]:
    if prev_parking and len(prev_parking) == k:
        return list(prev_parking)
    return [1] * k

def parking_zone(history_floors: np.ndarray, k: int, lobby_max: int, mid_max: int, F_min: int, F_max: int) -> list[int]:
    """
    Allocate cars across zones proportional to recent demand in each zone.
    Park each zone at its (unweighted) median floor in that zone (fallback to zone center).
    """
    if len(history_floors) == 0:
        # fallback: 50/30/20 across zones
        alloc = [math.ceil(0.5*k), math.ceil(0.3*k), k - math.ceil(0.5*k) - math.ceil(0.3*k)]
        targets = []
        targets += [1] * alloc[0]
        targets += [int((lobby_max + mid_max) / 2)] * alloc[1]
        targets += [int((mid_max + F_max) / 2)] * alloc[2]
        return targets

    hf = history_floors.astype(int)
    lob = hf[hf <= lobby_max]
    mid = hf[(hf > lobby_max) & (hf <= mid_max)]
    upp = hf[hf > mid_max]
    counts = np.array([len(lob), len(mid), len(upp)], dtype=float)
    if counts.sum() == 0:
        counts = np.array([1, 1, 1], dtype=float)

    # proportional allocation, at least 1 per non-empty zone if possible
    props = counts / counts.sum()
    alloc = np.floor(props * k).astype(int)
    # distribute remaining
    while alloc.sum() < k:
        idx = int(np.argmax(props - alloc / max(1, k)))
        alloc[idx] += 1
    # ensure nonnegative and sum==k
    alloc = np.maximum(alloc, 0)
    # medians / centers
    def med_or_center(arr, lo, hi):
        if len(arr) > 0:
            return int(np.median(arr))
        return int((lo + hi) / 2)

    p_lob = 1
    p_mid = med_or_center(mid, lobby_max+1, mid_max)
    p_upp = med_or_center(upp, mid_max+1, F_max)
    return [p_lob]*alloc[0] + [p_mid]*alloc[1] + [p_upp]*alloc[2]

def parking_dynamic(history_floors: np.ndarray, k: int, fallback: list[int]) -> list[int]:
    """
    A lightweight surrogate for 1D weighted k-median:
    choose k quantile floors from the recent empirical floor distribution.
    """
    if len(history_floors) == 0:
        return list(fallback)
    return weighted_quantile_floors(history_floors, k)

# -------------------------
# main evaluator
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", default=".", help="Repo root (contains data_cleaning/ and outputs/).")
    ap.add_argument("--hall_calls", default="data_cleaning/hall_calls_clean.csv", help="Cleaned hall calls CSV.")
    ap.add_argument("--k", type=int, default=8, help="Number of elevators (idle cars to park).")
    ap.add_argument("--v_floor", type=float, default=1.5, help="Seconds per floor (simulation).")
    ap.add_argument("--door", type=float, default=8.0, help="Door time (simulation).")
    ap.add_argument("--history_bins", type=int, default=12, help="How many 5-min bins to look back (12=1 hour).")
    ap.add_argument("--lobby_max", type=int, default=3, help="Max floor for Lobby zone.")
    ap.add_argument("--mid_max", type=int, default=8, help="Max floor for Mid zone.")
    ap.add_argument("--out_csv", default="outputs/task3_call_metrics.csv")
    args = ap.parse_args()

    repo = os.path.abspath(args.repo_root)
    hall_path = os.path.join(repo, args.hall_calls)
    out_path = os.path.join(repo, args.out_csv)
    ensure_dir(os.path.dirname(out_path))

    df = pd.read_csv(hall_path)
    if "Time" not in df.columns or "Floor" not in df.columns:
        raise ValueError("hall_calls_clean.csv must contain columns: Time, Floor (Direction optional).")
    df["Time"] = pd.to_datetime(df["Time"])
    df["Floor"] = df["Floor"].astype(int)
    if "Direction" not in df.columns:
        df["Direction"] = "Unknown"

    # 5-min bins
    df["bin5"] = df["Time"].dt.floor("5min")
    # interval stats
    g = df.groupby("bin5", sort=True)
    interval = g.agg(
        count=("Floor", "size"),
        up_count=("Direction", lambda s: int((s.astype(str).str.lower() == "up").sum())),
        floors=("Floor", lambda s: list(s.astype(int).values)),
        hour=("Time", lambda s: int(pd.to_datetime(s.iloc[0]).hour)),
        weekday=("Time", lambda s: int(pd.to_datetime(s.iloc[0]).weekday())),
    ).reset_index()

    interval["up_ratio"] = interval["up_count"] / interval["count"].clip(lower=1)
    # thresholds for state logic
    q25 = float(interval["count"].quantile(0.25))
    q75 = float(interval["count"].quantile(0.75))
    interval["is_weekday"] = interval["weekday"] <= 4
    interval["state"] = interval.apply(lambda r: compute_state(
        int(r["count"]), float(r["up_ratio"]), bool(r["is_weekday"]), int(r["hour"]), q25, q75
    ), axis=1)

    F_min = int(df["Floor"].min())
    F_max = int(df["Floor"].max())

    strategies = ["last_stop", "lobby", "zone", "dynamic"]
    logs = []

    # per-strategy parking memory
    prev_parking = {s: [1] * args.k for s in strategies}

    # rolling history of floors by bin index (for demand estimation)
    floors_by_bin = [np.array(x, dtype=int) for x in interval["floors"].tolist()]

    for i, row in interval.iterrows():
        bin_start = row["bin5"]
        state = row["state"]

        # history window [i-history_bins, i)
        j0 = max(0, i - args.history_bins)
        hist_floors = np.concatenate(floors_by_bin[j0:i]) if i > j0 else np.array([], dtype=int)

        for strat in strategies:
            prev = prev_parking[strat]
            if strat == "last_stop":
                curr = parking_last_stop(prev, args.k)
            elif strat == "lobby":
                curr = parking_lobby(args.k)
            elif strat == "zone":
                curr = parking_zone(hist_floors, args.k, args.lobby_max, args.mid_max, F_min, F_max)
            elif strat == "dynamic":
                # dynamic falls back to zone parking if no history
                fallback = parking_zone(hist_floors, args.k, args.lobby_max, args.mid_max, F_min, F_max)
                curr = parking_dynamic(hist_floors, args.k, fallback)
            else:
                raise ValueError(strat)

            # empty travel between parking floors (1D matching)
            empty_travel = match_sorted_distance(prev, curr)
            prev_parking[strat] = curr

            # compute call-level wait times in this bin
            calls = df[df["bin5"] == bin_start][["Time", "Floor"]]
            if len(calls) == 0:
                continue

            parks = np.array(curr, dtype=int)
            for _, c in calls.iterrows():
                f = int(c["Floor"])
                dist = int(np.min(np.abs(parks - f)))
                wait = dist * args.v_floor + args.door
                logs.append({
                    "strategy": strat,
                    "call_time": c["Time"],
                    "wait_time": float(wait),
                    "state": state,
                    "empty_travel": float(empty_travel) / float(len(calls))  # per-call share for convenience
                })

    out = pd.DataFrame(logs)
    out.to_csv(out_path, index=False)
    print(f"[OK] Wrote {out_path}")
    print(out.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
