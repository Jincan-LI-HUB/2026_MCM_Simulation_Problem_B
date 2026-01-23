import pandas as pd

# 1) 读入你们生成的 CSV
df = pd.read_csv("outputs/report_table_task1_by_state.csv")

# 2) 排序：样本多的 state 排前面（更可信）
df = df.sort_values("N", ascending=False)

# 3) 只展示前 8 个 state（太长会占页数）
df_show = df.head(8).copy()

# 4) 格式化（保留 3 位小数）
for col in ["AR1_MAE", "AR1_RMSE", "BASE_MAE", "BASE_RMSE"]:
    df_show[col] = df_show[col].map(lambda x: f"{x:.3f}")

df_show["N"] = df_show["N"].astype(int)

# 5) 输出 LaTeX 表格片段
latex = []
latex.append(r"\begin{table}[htbp]")
latex.append(r"\centering")
latex.append(r"\caption{Task 1 per-state benchmark (top states by sample size).}")
latex.append(r"\label{tab:task1-benchmark-by-state}")
latex.append(r"\begin{tabular}{lrrrrr}")
latex.append(r"\hline")
latex.append(r"State & $N$ & AR1 MAE & Base MAE & AR1 RMSE & Base RMSE \\")
latex.append(r"\hline")

for _, r in df_show.iterrows():
    latex.append(
        f"{r['state']} & {r['N']} & {r['AR1_MAE']} & {r['BASE_MAE']} & {r['AR1_RMSE']} & {r['BASE_RMSE']} \\\\"
    )

latex.append(r"\hline")
latex.append(r"\end{tabular}")
latex.append(r"\end{table}")

with open("outputs/latex_table_task1_by_state.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(latex))

print("[OK] Wrote outputs/latex_table_task1_by_state.tex")
