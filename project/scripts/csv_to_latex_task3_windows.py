import pandas as pd
import os

IN_CSV = "outputs/report_table_task3_windows.csv"
OUT_TEX = "outputs/latex_table_task3_windows.tex"

# 1) 读入 windows CSV
df = pd.read_csv(IN_CSV)

# 2) 兼容字段名（不同版本可能略有不同）
required = ["window", "strategy", "AWT", "P95", "P99", "LongWait%", "N"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column '{c}' in {IN_CSV}. Found: {list(df.columns)}")

# 3) 排序：先按 window，再按 AWT
df = df.sort_values(["window", "AWT"], ascending=[True, True]).copy()

# 4) 格式化数字
df["N"] = df["N"].astype(int)
df["AWT"] = df["AWT"].map(lambda x: f"{x:.2f}")
df["P95"] = df["P95"].map(lambda x: f"{x:.2f}")
df["P99"] = df["P99"].map(lambda x: f"{x:.2f}")
df["LongWait%"] = df["LongWait%"].map(lambda x: f"{x:.2f}")

# 5) 生成 LaTeX（按 window 分组，用 \multicolumn 做小标题）
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"\centering")
lines.append(r"\caption{Task 3 performance under critical operating windows.}")
lines.append(r"\label{tab:task3-windows}")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\hline")
lines.append(r"Strategy & AWT & P95 & P99 & Long wait (\%) & $N$ \\")
lines.append(r"\hline")

for w, g in df.groupby("window", sort=False):
    # window header row
    safe_w = str(w).replace("&", r"\&")
    lines.append(rf"\multicolumn{{6}}{{l}}{{\textbf{{{safe_w}}}}} \\")
    for _, r in g.iterrows():
        strat = str(r["strategy"]).replace("_", r"\_")
        lines.append(
            f"{strat} & {r['AWT']} & {r['P95']} & {r['P99']} & {r['LongWait%']} & {r['N']} \\\\"
        )
    lines.append(r"\hline")

lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

# 6) 输出
os.makedirs("outputs", exist_ok=True)
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"[OK] Wrote {OUT_TEX}")
