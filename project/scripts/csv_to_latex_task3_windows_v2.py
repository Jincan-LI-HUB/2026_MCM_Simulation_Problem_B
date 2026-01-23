#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/csv_to_latex_task3_windows_v2.py

Upgrades over your v1:
- Enforces window order: Peak -> Transition -> High-load
- Adds EmptyTravel_mean column when present
- Shows the long-wait threshold used per window if present (LongWaitThreshold)
"""

import pandas as pd
import os

IN_CSV = "outputs/report_table_task3_windows.csv"
OUT_TEX = "outputs/latex_table_task3_windows.tex"

WINDOW_ORDER = [
    "Peak (weekday 08-10,17-19)",
    "Transition (±15min around label changes)",
    "High-load (top 10% 5-min bins)",
]

df = pd.read_csv(IN_CSV)

required = ["window", "strategy", "AWT", "P95", "P99", "LongWait%", "N"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in {IN_CSV}. Found: {list(df.columns)}")

has_empty = "EmptyTravel_mean" in df.columns
has_thr = "LongWaitThreshold" in df.columns

df["N"] = df["N"].astype(int)

# numeric formatting
for col, fmt in [("AWT","{:.2f}"), ("P95","{:.2f}"), ("P99","{:.2f}"), ("LongWait%","{:.2f}")]:
    df[col] = df[col].map(lambda x: fmt.format(float(x)))

if has_empty:
    df["EmptyTravel_mean"] = df["EmptyTravel_mean"].map(lambda x: "{:.2f}".format(float(x)) if pd.notna(x) else "")

if has_thr:
    df["LongWaitThreshold"] = df["LongWaitThreshold"].map(lambda x: "{:.2f}".format(float(x)) if pd.notna(x) else "")

# sort within each window by AWT ascending
df["AWT_num"] = df["AWT"].astype(float)
df = df.sort_values(["window", "AWT_num"], ascending=[True, True]).drop(columns=["AWT_num"])

# LaTeX
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Task 3 performance under operationally critical windows.}")
lines.append(r"\label{tab:task3-windows}")

# column spec and header
if has_empty:
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Strategy & AWT & P95 & P99 & Long wait (\%) & Empty travel & $N$ \\")
else:
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\hline")
    lines.append(r"Strategy & AWT & P95 & P99 & Long wait (\%) & $N$ \\")

lines.append(r"\hline")

def escape_tex(s: str) -> str:
    return str(s).replace("&", r"\&").replace("_", r"\_")

# enforce window order, then any extras at end
seen = set(df["window"].unique().tolist())
ordered = [w for w in WINDOW_ORDER if w in seen] + [w for w in seen if w not in WINDOW_ORDER]

for w in ordered:
    g = df[df["window"] == w]
    if len(g) == 0:
        continue
    w_title = escape_tex(w)
    if has_thr:
        thr_vals = g["LongWaitThreshold"].dropna().unique().tolist()
        thr_note = f" (long-wait threshold: {thr_vals[0]} )" if len(thr_vals) > 0 and thr_vals[0] != "" else ""
        lines.append(rf"\multicolumn{{{7 if has_empty else 6}}}{{l}}{{\textbf{{{w_title}{escape_tex(thr_note)}}}}} \\")
    else:
        lines.append(rf"\multicolumn{{{7 if has_empty else 6}}}{{l}}{{\textbf{{{w_title}}}}} \\")
    for _, r in g.iterrows():
        strat = escape_tex(r["strategy"])
        if has_empty:
            lines.append(f"{strat} & {r['AWT']} & {r['P95']} & {r['P99']} & {r['LongWait%']} & {r['EmptyTravel_mean']} & {r['N']} \\\\")
        else:
            lines.append(f"{strat} & {r['AWT']} & {r['P95']} & {r['P99']} & {r['LongWait%']} & {r['N']} \\\\")
    lines.append(r"\hline")

lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

os.makedirs("outputs", exist_ok=True)
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[OK] Wrote {OUT_TEX}")
