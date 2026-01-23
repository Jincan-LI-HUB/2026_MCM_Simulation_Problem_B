#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run 3 stress tests for Task 3 and generate:
- outputs/tab/task3_strategy_comparison.tex  (Table 4: AWT + P95 + LongWait%)
- outputs/tab/appendixF_stress.tex           (Appendix F: stress-test summary)
- outputs/data/stress_test_summary.csv       (raw summary)

Usage (from project root):
    python scripts/run_stress_tests.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# Paths (match your project)
# -----------------------------
ROOT = Path.cwd()
OUT_DIR = ROOT / "outputs"
OUT_DATA = OUT_DIR / "data"
OUT_TAB  = OUT_DIR / "tab"
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)

PATH_HALL  = ROOT / "data" / "clean" / "hall_calls_clean.csv"
PATH_STOP  = ROOT / "data" / "clean" / "car_stops_clean.csv"
PATH_DEP   = ROOT / "data" / "clean" / "car_departures_clean.csv"
PATH_LOAD  = ROOT / "data" / "clean" / "load_changes_clean.csv"
PATH_MAINT = ROOT / "data" / "clean" / "maintenance_mode_clean.csv"

# -----------------------------
# Config (must match notebook)
# -----------------------------
FREQ = "5min"
DECISION_FREQ = "5min"
PRED_HORIZON_MIN = 15

N_ELEVATORS = 8
LOBBY_FLOOR = 1

SECONDS_PER_FLOOR_BASE = 1.5
DOOR_TIME_BASE = 8.0
LONG_WAIT_THRESHOLD_BASE = 60.0

N_CLUSTERS = 6
RNG_SEED = 42


def build_5min_from_hall_calls(hall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build minimal 5-min features for rule-based mode classification
    using ONLY hall call events.
    Requires: hall_df has columns Time, Floor, and optionally Direction.
    Output columns: Time(slice), count, up_ratio, hour, is_weekend
    """
    df = hall_df.copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df["slice"] = df["Time"].dt.floor("5min")

    # total count
    g = df.groupby("slice").size().rename("count").reset_index()

    # direction ratio (optional)
    if "Direction" in df.columns:
        tmp = df.groupby(["slice", "Direction"]).size().unstack(fill_value=0)
        up = tmp.get("Up", 0)
        down = tmp.get("Down", 0)
        denom = (up + down).replace(0, np.nan)
        up_ratio = (up / denom).fillna(0.5)
        g = g.merge(up_ratio.rename("up_ratio").reset_index(), on="slice", how="left")
    else:
        g["up_ratio"] = 0.5

    g = g.rename(columns={"slice": "Time"}).sort_values("Time").reset_index(drop=True)
    g["hour"] = g["Time"].dt.hour
    g["is_weekend"] = (g["Time"].dt.weekday >= 5).astype(int)
    return g

def compute_theta1(count_series: pd.Series, q: float = 0.10) -> float:
    nonzero = count_series[count_series > 0]
    if len(nonzero) >= 30:
        theta = float(nonzero.quantile(q))
    else:
        theta = 1.0
    return max(1.0, theta)

def classify_rule_mode_from_minfeat(row: pd.Series, theta1: float) -> str:
    C = float(row["count"])
    up = float(row["up_ratio"])
    hour = int(row["hour"])
    is_weekend = int(row["is_weekend"])

    # Night/Idle fix (inclusive)
    if (hour <= 6 or hour >= 22) and (C <= theta1):
        return "Idle/Low"

    if is_weekend == 1 and C <= max(theta1 + 1.0, 2.0):
        return "Idle/Low"

    if C >= 6 and up >= 0.65:
        return "Up-Peak"
    if C >= 6 and up <= 0.35:
        return "Down-Peak"

    if C >= 6 and 0.35 < up < 0.65:
        return "Mixed"

    return "Normal"


# -----------------------------
# Utilities
# -----------------------------
def pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def write_booktabs_table(path_tex: Path, caption: str, label: str, headers, rows, colfmt=None):
    def esc(x):
        if isinstance(x, str):
            return x.replace("_", r"\_")
        return str(x)

    if colfmt is None:
        colfmt = "l" + "r" * (len(headers) - 1)

    with open(path_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\\centering\n")
        f.write(f"\\caption{{{esc(caption)}}}\n")
        f.write(f"\\label{{{esc(label)}}}\n")
        f.write(f"\\begin{{tabular}}{{{colfmt}}}\n\\toprule\n")
        f.write(" & ".join(map(esc, headers)) + " \\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(esc(v) for v in r) + " \\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# -----------------------------
# Data loading
# -----------------------------
def load_data():
    hall = pd.read_csv(PATH_HALL)
    stop = pd.read_csv(PATH_STOP) if PATH_STOP.exists() else pd.DataFrame()
    dep  = pd.read_csv(PATH_DEP) if PATH_DEP.exists() else pd.DataFrame()
    load = pd.read_csv(PATH_LOAD) if PATH_LOAD.exists() else pd.DataFrame()
    maint= pd.read_csv(PATH_MAINT) if PATH_MAINT.exists() else pd.DataFrame()
    # -------- Route B: build rule timeline from hall calls only --------
    minfeat = build_5min_from_hall_calls(hall_df)
    theta1 = compute_theta1(minfeat["count"], q=0.10)
    minfeat["mode_rule"] = minfeat.apply(lambda r: classify_rule_mode_from_minfeat(r, theta1), axis=1)

    rule_tl = minfeat[["Time", "mode_rule"]].copy()
    rule_tl.to_csv("outputs/data/task2_modes_timeline_rule.csv", index=False)
    # normalize time
    for df in [hall, stop, dep, load, maint]:
        if not df.empty:
            if "Time" not in df.columns:
                tcol = pick_col(df, ["time","timestamp","DateTime"], required=True)
                df.rename(columns={tcol:"Time"}, inplace=True)
            df["Time"] = pd.to_datetime(df["Time"])
            df.sort_values("Time", inplace=True)

    return hall, stop, dep, load, maint


# -----------------------------
# Feature engineering (lightweight)
# -----------------------------
def build_5min_features(hall: pd.DataFrame, dep: pd.DataFrame, load: pd.DataFrame, maint: pd.DataFrame) -> pd.DataFrame:
    h_floor = pick_col(hall, ["Floor","floor","OriginFloor","FromFloor","HallFloor","StartFloor"], required=True)
    h = hall[["Time", h_floor]].copy()
    h.rename(columns={h_floor:"Floor"}, inplace=True)

    h["Slice"] = h["Time"].dt.floor(FREQ)
    calls = h.groupby("Slice").size().rename("hall_calls").to_frame()

    dep_cnt = None
    if not dep.empty:
        dep2 = dep[["Time"]].copy()
        dep2["Slice"] = dep2["Time"].dt.floor(FREQ)
        dep_cnt = dep2.groupby("Slice").size().rename("departures").to_frame()

    net_load = None
    if not load.empty:
        dcol = pick_col(load, ["DeltaLoad","delta_load","LoadChange","NetLoad","dLoad"], required=False)
        if dcol is not None:
            l2 = load[["Time", dcol]].copy()
            l2["Slice"] = l2["Time"].dt.floor(FREQ)
            net_load = l2.groupby("Slice")[dcol].sum().rename("net_load").to_frame()

    maint_ratio = None
    if not maint.empty:
        mcol = pick_col(maint, ["IsMaint","Maintenance","Maint","Status"], required=False)
        if mcol is not None:
            m2 = maint[["Time", mcol]].copy()
            m2["Slice"] = m2["Time"].dt.floor(FREQ)
            m2["is_maint"] = m2[mcol].astype(int) if np.issubdtype(m2[mcol].dtype, np.number) else m2[mcol].astype(str).str.lower().isin(["1","true","yes","y","maint"])
            maint_ratio = m2.groupby("Slice")["is_maint"].mean().rename("maint_ratio").to_frame()

    dir_col = pick_col(hall, ["Direction","direction","Dir"], required=False)
    up_ratio = None
    if dir_col is not None:
        dd = hall[["Time", dir_col]].copy()
        dd["Slice"] = dd["Time"].dt.floor(FREQ)
        s = dd[dir_col].astype(str).str.lower()
        is_up = s.str.contains("up") | s.str.fullmatch("u") | s.str.contains("↑")
        is_dn = s.str.contains("down") | s.str.fullmatch("d") | s.str.contains("↓")
        tmp = pd.DataFrame({"Slice": dd["Slice"], "up": is_up.astype(int), "dn": is_dn.astype(int)})
        g = tmp.groupby("Slice")[["up","dn"]].sum()
        up_ratio = (g["up"] / (g["up"] + g["dn"] + 1e-9)).rename("up_ratio").to_frame()

    ent = h.groupby(["Slice","Floor"]).size().rename("n").reset_index()
    def _entropy(x):
        p = x / (x.sum() + 1e-12)
        return float(-(p * np.log(p + 1e-12)).sum())
    entropy = ent.groupby("Slice")["n"].apply(_entropy).rename("entropy").to_frame()

    feat = calls.join(entropy, how="outer")
    if up_ratio is not None:
        feat = feat.join(up_ratio, how="left")
    if dep_cnt is not None:
        feat = feat.join(dep_cnt, how="left")
    if net_load is not None:
        feat = feat.join(net_load, how="left")
    if maint_ratio is not None:
        feat = feat.join(maint_ratio, how="left")

    feat = feat.fillna(0.0).reset_index().rename(columns={"Slice":"Time"})

    for c in ["hall_calls","up_ratio","entropy","departures","net_load","maint_ratio"]:
        if c in feat.columns:
            feat[c+"_w"] = feat[c].rolling(3, min_periods=1).mean()

    return feat

def get_rule_label_at(t: pd.Timestamp, rule_timeline: pd.DataFrame) -> str:
    arr = rule_timeline["Time"].values
    idx = np.searchsorted(arr, np.datetime64(t), side="right") - 1
    if idx < 0:
        return "Normal"
    return str(rule_timeline.iloc[idx]["mode_rule"])


# -----------------------------
# Mode clustering + labeling
# -----------------------------
def train_mode_cluster(feat: pd.DataFrame, n_clusters=N_CLUSTERS):
    cand = ["hall_calls_w","up_ratio_w","entropy_w","departures_w","net_load_w","maint_ratio_w",
            "hall_calls","up_ratio","entropy","departures","net_load","maint_ratio"]
    cols = [c for c in cand if c in feat.columns]
    X = feat[cols].values
    pipe = Pipeline([("scaler", StandardScaler()), ("kmeans", KMeans(n_clusters=n_clusters, random_state=RNG_SEED, n_init=10))])
    pipe.fit(X)
    out = feat.copy()
    out["cluster"] = pipe.predict(X)
    return pipe, cols, out

def label_clusters(clustered_feat: pd.DataFrame):
    grp = clustered_feat.groupby("cluster")
    hall = grp["hall_calls_w"].mean() if "hall_calls_w" in clustered_feat.columns else grp["hall_calls"].mean()
    up = grp["up_ratio_w"].mean() if "up_ratio_w" in clustered_feat.columns else (grp["up_ratio"].mean() if "up_ratio" in clustered_feat.columns else pd.Series(0.5, index=hall.index))
    hall_q1, hall_q3 = np.quantile(hall.values, 0.25), np.quantile(hall.values, 0.75)
    mapping = {}
    for k in hall.index:
        if hall[k] >= hall_q3 and up[k] >= 0.6:
            mapping[int(k)] = "Up-Peak"
        elif hall[k] >= hall_q3 and up[k] <= 0.4:
            mapping[int(k)] = "Down-Peak"
        elif hall[k] <= hall_q1:
            mapping[int(k)] = "Idle/Low"
        else:
            mapping[int(k)] = "Mixed"
    return mapping


# -----------------------------
# Demand by mode (cluster, floor)
# -----------------------------
def learn_floor_demand_by_mode(hall: pd.DataFrame, feats_clustered: pd.DataFrame):
    h_floor = pick_col(hall, ["Floor","floor","OriginFloor","FromFloor","HallFloor","StartFloor"], required=True)
    hall2 = hall[["Time", h_floor]].copy()
    hall2.rename(columns={h_floor:"floor"}, inplace=True)
    hall2["Time"] = pd.to_datetime(hall2["Time"])
    hall2["Slice"] = hall2["Time"].dt.floor(FREQ)

    feat_idx = feats_clustered[["Time","cluster"]].copy()
    feat_idx["Time"] = pd.to_datetime(feat_idx["Time"])
    feat_idx.rename(columns={"Time":"Slice"}, inplace=True)

    merged = hall2.merge(feat_idx, on="Slice", how="left")
    merged["cluster"] = merged["cluster"].ffill().fillna(0).astype(int)
    return merged.groupby(["cluster","floor"]).size().rename("cnt").reset_index()


# -----------------------------
# Simulator (P95 included)
# -----------------------------
class ElevatorState:
    def __init__(self, floor: int, available_time: pd.Timestamp):
        self.floor = int(floor)
        self.available_time = available_time

def travel_time_seconds(f1: int, f2: int, seconds_per_floor: float):
    return abs(int(f1) - int(f2)) * float(seconds_per_floor)

def infer_initial_positions(stop: pd.DataFrame, start_time: pd.Timestamp):
    eid_col = pick_col(stop, ["Elevator","elevator","CarID","CarId","LiftID","ID"], required=False)
    floor_col = pick_col(stop, ["Floor","floor","StopFloor","AtFloor","CurrentFloor"], required=False)
    pos = {i: LOBBY_FLOOR for i in range(N_ELEVATORS)}
    if stop.empty or eid_col is None or floor_col is None:
        return pos
    s = stop[stop["Time"] <= start_time].sort_values("Time")
    if s.empty:
        return pos
    last = s.groupby(eid_col).tail(1)
    for _, r in last.iterrows():
        try:
            pos[int(r[eid_col])] = int(r[floor_col])
        except Exception:
            pass
    return pos

def simulate(strategy: str,
             hall: pd.DataFrame,
             stop: pd.DataFrame,
             feats: pd.DataFrame,
             mode_pipe: Pipeline,
             mode_features,
             demand_by_cluster_floor: pd.DataFrame,
             start_time: pd.Timestamp,
             end_time: pd.Timestamp,
             seconds_per_floor: float,
             door_time: float,
             long_wait_threshold: float):

    h_floor = pick_col(hall, ["Floor","floor","OriginFloor","FromFloor","HallFloor","StartFloor"], required=True)
    hall = hall.sort_values("Time").copy()
    hall_sim = hall[(hall["Time"] >= start_time) & (hall["Time"] <= end_time)].copy()
    if hall_sim.empty:
        return {"AWT": np.nan, "P95": np.nan, "LongWaitPct": np.nan, "N": 0}

    init_pos = infer_initial_positions(stop, start_time)
    elevators = {e: ElevatorState(floor=init_pos.get(e, LOBBY_FLOOR), available_time=start_time) for e in range(N_ELEVATORS)}

    demand_lookup = {}
    for cl, dfc in demand_by_cluster_floor.groupby("cluster"):
        floors = dfc["floor"].astype(int).values
        w = dfc["cnt"].astype(float).values
        if w.sum() <= 0:
            w = np.ones_like(w)
        demand_lookup[int(cl)] = (floors, w)

    feats_idx = feats.set_index("Time").sort_index()

    def get_cluster_at(t: pd.Timestamp):
        tt = t.floor(DECISION_FREQ)
        if tt in feats_idx.index:
            row = feats_idx.loc[tt]
        else:
            row = feats_idx.loc[:tt].tail(1)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
        X = row[mode_features].fillna(0.0).values.reshape(1, -1)
        return int(mode_pipe.predict(X)[0])

    def apply_parking_policy(t: pd.Timestamp):
        idle = [e for e, st in elevators.items() if st.available_time <= t]
        if not idle:
            return

        if strategy == "last_stop":
            return

        if strategy == "lobby":
            for e in idle:
                st = elevators[e]
                if st.floor != LOBBY_FLOOR:
                    tt = travel_time_seconds(st.floor, LOBBY_FLOOR, seconds_per_floor)
                    st.available_time = t + pd.Timedelta(seconds=tt)
                    st.floor = LOBBY_FLOOR
            return

        if strategy == "dynamic":
            cl = get_cluster_at(t)
            if cl in demand_lookup:
                floors, w = demand_lookup[cl]
            else:
                floors = np.arange(int(hall_sim[h_floor].min()), int(hall_sim[h_floor].max()) + 1)
                w = np.ones_like(floors, dtype=float)

            k = len(idle)
            order = np.argsort(floors)
            floors2, w2 = floors[order], w[order]
            cw = np.cumsum(w2) / (w2.sum() + 1e-12)
            qs = [(i + 0.5) / k for i in range(k)]
            targets = [int(floors2[np.searchsorted(cw, q)]) for q in qs]

            for e, tgt in zip(idle, targets):
                st = elevators[e]
                if st.floor != tgt:
                    tt = travel_time_seconds(st.floor, tgt, seconds_per_floor)
                    st.available_time = t + pd.Timedelta(seconds=tt)
                    st.floor = tgt

    decision_times = pd.date_range(start=start_time.floor(DECISION_FREQ),
                                   end=end_time.ceil(DECISION_FREQ),
                                   freq=DECISION_FREQ)
    dt_iter = iter(decision_times)
    next_dec = next(dt_iter, None)

    waits = []
    for _, r in hall_sim.iterrows():
        call_time = r["Time"]
        origin_floor = int(r[h_floor])

        while next_dec is not None and next_dec <= call_time:
            apply_parking_policy(next_dec)
            next_dec = next(dt_iter, None)

        best_e, best_arrival = None, None
        for e, st in elevators.items():
            start_service = max(st.available_time, call_time)
            arr = start_service + pd.Timedelta(seconds=travel_time_seconds(st.floor, origin_floor, seconds_per_floor))
            if best_arrival is None or arr < best_arrival:
                best_arrival = arr
                best_e = e

        waits.append(float((best_arrival - call_time).total_seconds()))
        st = elevators[best_e]
        st.floor = origin_floor
        st.available_time = best_arrival + pd.Timedelta(seconds=float(door_time))

    waits = np.asarray(waits, dtype=float)
    return {
        "AWT": float(np.mean(waits)),
        "P95": float(np.percentile(waits, 95)),
        "LongWaitPct": float(np.mean(waits >= float(long_wait_threshold)) * 100.0),
        "N": int(len(waits)),
    }


# -----------------------------
# Stress scenarios
# -----------------------------
def scale_hall(hall_sim: pd.DataFrame, factor: float, rng: np.random.Generator):
    if factor <= 1.0:
        n = max(1, int(len(hall_sim) * factor))
        return hall_sim.sample(n=n, replace=False, random_state=RNG_SEED).sort_values("Time")
    extra = int(len(hall_sim) * (factor - 1.0))
    add = hall_sim.sample(n=extra, replace=True, random_state=RNG_SEED).copy()
    add["Time"] = add["Time"] + pd.to_timedelta(rng.integers(0, 60, size=len(add)), unit="s")
    return pd.concat([hall_sim, add], ignore_index=True).sort_values("Time")

def inject_shock(hall_sim: pd.DataFrame, shock_floor: int, burst_calls: int, burst_minutes: int):
    tmp = hall_sim.copy()
    tmp["hour"] = tmp["Time"].dt.floor("h")
    counts = tmp.groupby("hour").size()
    hour0 = counts.idxmin()
    start = hour0 + pd.Timedelta(minutes=5)
    end = start + pd.Timedelta(minutes=burst_minutes)
    times = pd.date_range(start=start, end=end, periods=burst_calls)

    fcol = pick_col(hall_sim, ["Floor","floor","OriginFloor","FromFloor","HallFloor","StartFloor"], required=True)
    add = pd.DataFrame({"Time": times, fcol: shock_floor})
    return pd.concat([hall_sim, add], ignore_index=True).sort_values("Time")


def main():
    rng = np.random.default_rng(RNG_SEED)
    hall, stop, dep, load, maint = load_data()

    feats = build_5min_features(hall, dep, load, maint)
    mode_pipe, mode_features, feats_clustered = train_mode_cluster(feats, n_clusters=N_CLUSTERS)
    demand = learn_floor_demand_by_mode(hall, feats_clustered)

    tmin, tmax = hall["Time"].min(), hall["Time"].max()
    sim_start = max(tmin, tmax - pd.Timedelta(days=3))
    sim_end = tmax
    hall_sim_base = hall[(hall["Time"] >= sim_start) & (hall["Time"] <= sim_end)].copy()

    scenarios = [
        ("base", hall_sim_base, SECONDS_PER_FLOOR_BASE, DOOR_TIME_BASE, LONG_WAIT_THRESHOLD_BASE),
        ("scale_1p2", scale_hall(hall_sim_base, 1.2, rng), SECONDS_PER_FLOOR_BASE, DOOR_TIME_BASE, LONG_WAIT_THRESHOLD_BASE),
        ("scale_1p5", scale_hall(hall_sim_base, 1.5, rng), SECONDS_PER_FLOOR_BASE, DOOR_TIME_BASE, LONG_WAIT_THRESHOLD_BASE),
        ("shock_burst", inject_shock(hall_sim_base, LOBBY_FLOOR, burst_calls=120, burst_minutes=10), SECONDS_PER_FLOOR_BASE, DOOR_TIME_BASE, LONG_WAIT_THRESHOLD_BASE),
        ("params_plus20", hall_sim_base, SECONDS_PER_FLOOR_BASE * 1.2, DOOR_TIME_BASE * 1.2, LONG_WAIT_THRESHOLD_BASE),
    ]
    strategies = ["last_stop", "lobby", "dynamic", "dynamic_kmeans", "dynamic_rule"]

    out_rows = []
    for scen, hsim, spf, door, thr in scenarios:
        for strat in strategies:
            res = simulate(
                strategy=strat,
                hall=hsim,
                stop=stop,
                feats=feats_clustered,
                mode_pipe=mode_pipe,
                mode_features=mode_features,
                demand_by_cluster_floor=demand,
                start_time=sim_start,
                end_time=sim_end,
                seconds_per_floor=spf,
                door_time=door,
                long_wait_threshold=thr,
            )
            out_rows.append({
                "scenario": scen,
                "strategy": strat,
                "AWT": res["AWT"],
                "P95": res["P95"],
                "LongWaitPct": res["LongWaitPct"],
                "N": res["N"],
            })

    summary = pd.DataFrame(out_rows)
    summary.to_csv(OUT_DATA / "stress_test_summary.csv", index=False)
    print("Saved:", OUT_DATA / "stress_test_summary.csv")

    # Table 4 (base)
    base = summary[summary["scenario"] == "base"].copy()
    base = base.sort_values("strategy")
    main_rows = [[r["strategy"], f"{r['AWT']:.2f}", f"{r['P95']:.2f}", f"{r['LongWaitPct']:.2f}"] for _, r in base.iterrows()]
    write_booktabs_table(
        OUT_TAB / "task3_strategy_comparison.tex",
        caption="Simulation comparison of parking strategies.",
        label="tab:task3-strategy",
        headers=["Strategy", "AWT", "P95", r"Long wait (\%)"],
        rows=main_rows,
        colfmt="lrrr",
    )
    print("Wrote:", OUT_TAB / "task3_strategy_comparison.tex")

    # Appendix F (compact): show last_stop and dynamic only
    keep = summary[summary["strategy"].isin(["last_stop", "dynamic"])].copy()
    keep = keep.sort_values(["scenario", "strategy"])
    app_rows = [[r["scenario"], r["strategy"], f"{r['AWT']:.2f}", f"{r['P95']:.2f}", f"{r['LongWaitPct']:.2f}"] for _, r in keep.iterrows()]
    write_booktabs_table(
        OUT_TAB / "appendixF_stress.tex",
        caption="Stress-test results using tail-risk aware metrics (AWT, P95, and long-wait rate).",
        label="tab:stress-tests",
        headers=["Scenario", "Strategy", "AWT", "P95", r"Long wait (\%)"],
        rows=app_rows,
        colfmt="llrrr",
    )
    print("Wrote:", OUT_TAB / "appendixF_stress.tex")

    print("Done. Recompile LaTeX to refresh Table 4 and Appendix F.")

if __name__ == "__main__":
    main()
