#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd


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


def infer_time_col(df: pd.DataFrame):
    for c in ["Time", "time", "timestamp", "Datetime", "datetime"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find time column.")


def infer_floor_col(df: pd.DataFrame):
    for c in ["Floor", "floor", "FromFloor", "from_floor", "OriginFloor", "origin_floor"]:
        if c in df.columns:
            return c
    raise ValueError("Cannot find floor column.")


def travel_time_seconds(f1: int, f2: int, sec_per_floor: float) -> float:
    return abs(int(f1) - int(f2)) * float(sec_per_floor)


def simulate_baselines(hall_calls_path: str,
                       n_elevators: int = 6,
                       lobby_floor: int = 1,
                       sec_per_floor: float = 2.0,
                       door_time: float = 6.0,
                       long_wait_threshold: float = 60.0):
    """
    A lightweight, MCM-style comparative simulator:
    - dispatch: each call served by elevator with earliest arrival (availability + travel)
    - service time: travel + door_time (no in-car passenger routing, to keep simulator minimal and comparable)
    - baselines: last_stop, lobby_return
    This is not a full elevator physics model; it is a consistent comparative evaluator.
    """
    df = pd.read_csv(hall_calls_path)
    tcol = infer_time_col(df)
    fcol = infer_floor_col(df)
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.sort_values(tcol).reset_index(drop=True)

    # Represent each elevator by (available_time, current_floor)
    def run(strategy: str):
        avail = np.array([df[tcol].min().value] * n_elevators, dtype=np.int64)  # ns
        pos = np.array([lobby_floor] * n_elevators, dtype=np.int64)

        waits = []
        empty_travel = 0.0

        for _, r in df.iterrows():
            call_time_ns = int(r[tcol].value)
            call_floor = int(r[fcol])

            # reposition policy for idle elevators (very simplified):
            # if elevator is idle before call time, it may have been moved by strategy
            if strategy == "lobby":
                # send any elevator that is idle before call time to lobby (assume it happens immediately when idle)
                for i in range(n_elevators):
                    if avail[i] <= call_time_ns and pos[i] != lobby_floor:
                        empty_travel += abs(pos[i] - lobby_floor)
                        pos[i] = lobby_floor

            # choose elevator with min arrival time
            best_i = None
            best_arrive = None
            for i in range(n_elevators):
                start_ns = max(avail[i], call_time_ns)
                move_sec = travel_time_seconds(pos[i], call_floor, sec_per_floor)
                arrive_ns = start_ns + int(move_sec * 1e9)
                if best_arrive is None or arrive_ns < best_arrive:
                    best_arrive = arrive_ns
                    best_i = i

            wait_sec = (best_arrive - call_time_ns) / 1e9
            waits.append(wait_sec)

            # update chosen elevator state
            empty_travel += abs(pos[best_i] - call_floor)
            pos[best_i] = call_floor
            avail[best_i] = best_arrive + int(door_time * 1e9)

        waits = np.array(waits, dtype=float)
        return {
            "strategy": strategy,
            "AWT_sec": float(waits.mean()) if len(waits) else 0.0,
            "LongWait_pct": float((waits >= long_wait_threshold).mean() * 100.0) if len(waits) else 0.0,
            "EmptyTravel_floors": float(empty_travel),
            "n_calls": int(len(waits))
        }

    out = [run("last_stop"), run("lobby")]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", default="data/clean/hall_calls_clean.csv")
    ap.add_argument("--result_csv", default="outputs/data/task3_sim_results.csv",
                    help="If exists, read it; otherwise generate baseline results and save.")
    ap.add_argument("--out_tab", default="outputs/tab")
    ap.add_argument("--n_elevators", type=int, default=6)
    ap.add_argument("--sec_per_floor", type=float, default=2.0)
    ap.add_argument("--door_time", type=float, default=6.0)
    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.result_csv))
    ensure_dir(args.out_tab)

    if os.path.exists(args.result_csv):
        res = pd.read_csv(args.result_csv)
    else:
        # Baselines only; your "dynamic" should come from your unified notebook simulation.
        res = simulate_baselines(
            hall_calls_path=args.hall,
            n_elevators=args.n_elevators,
            sec_per_floor=args.sec_per_floor,
            door_time=args.door_time
        )
        res.to_csv(args.result_csv, index=False)

    # Create LaTeX table (expect dynamic row present; if not, table still compiles)
    needed_cols = ["strategy", "AWT_sec", "LongWait_pct"]
    for c in needed_cols:
        if c not in res.columns:
            raise ValueError(f"Missing column in results: {c}")

    # Format rows
    rows = []
    for _, r in res.iterrows():
        rows.append([
            str(r["strategy"]),
            f"{float(r['AWT_sec']):.2f}",
            f"{float(r['LongWait_pct']):.2f}",
        ])

    out_tex = os.path.join(args.out_tab, "task3_strategy_comparison.tex")
    write_booktabs_table(
        out_tex,
        caption="Simulation comparison of parking strategies.",
        label="tab:task3-strategy",
        headers=["Strategy", "AWT (s)", "Long wait (\%)"],
        rows=rows,
        colfmt="lrr"
    )

    print("Task3 materials generated:")
    print(" -", args.result_csv)
    print(" -", out_tex)
    print("Note: If you want the dynamic strategy row, export your unified notebook results to task3_sim_results.csv.")


if __name__ == "__main__":
    main()
