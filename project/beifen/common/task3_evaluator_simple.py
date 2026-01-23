# scripts/task3_evaluator_simple.py

import pandas as pd
import numpy as np
import os

# =========================
# 1. 参数（和论文一致）
# =========================
V_FLOOR = 1.5     # seconds per floor
DOOR_TIME = 8.0   # seconds
K_ELEVATORS = 8   # number of cars

# =========================
# 2. 读取 hall calls
# =========================
hall = pd.read_csv("data_cleaning/hall_calls_clean.csv")
hall["Time"] = pd.to_datetime(hall["Time"])

# 只保留必要字段
hall = hall[["Time", "Floor"]].sort_values("Time")

# =========================
# 3. 初始化 elevator 状态
# =========================
# 简化假设：所有 elevator 初始在 lobby (floor 1)
elevator_pos = np.ones(K_ELEVATORS, dtype=int)

# =========================
# 4. 不同 parking 策略
# =========================
def choose_parking_floor(strategy, call_floor):
    if strategy == "last_stop":
        return None  # 不 reposition
    if strategy == "lobby":
        return 1
    if strategy == "zone":
        if call_floor <= 3:
            return 1
        elif call_floor <= 8:
            return 6
        else:
            return 12
    if strategy == "dynamic":
        # 这里你可以后面接你们的 k-median 结果
        return call_floor
    raise ValueError(strategy)

# =========================
# 5. 主评估循环
# =========================
logs = []

for strategy in ["last_stop", "lobby", "zone", "dynamic"]:

    elevator_pos[:] = 1  # reset

    for _, row in hall.iterrows():
        call_time = row["Time"]
        call_floor = int(row["Floor"])

        # 找最近的 elevator
        dists = np.abs(elevator_pos - call_floor)
        car_id = np.argmin(dists)
        travel = dists[car_id] * V_FLOOR

        wait_time = travel + DOOR_TIME

        logs.append({
            "strategy": strategy,
            "call_time": call_time,
            "wait_time": wait_time
        })

        # 更新 elevator 位置（假设服务完停在 call_floor）
        elevator_pos[car_id] = call_floor

# =========================
# 6. 输出 CSV
# =========================
df = pd.DataFrame(logs)
os.makedirs("outputs", exist_ok=True)
df.to_csv("outputs/task3_call_metrics.csv", index=False)

print("[OK] Wrote outputs/task3_call_metrics.csv")
