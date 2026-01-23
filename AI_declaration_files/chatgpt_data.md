**ChatGPT Query 2:**
---
**hall_calls.csv**  
☑ Convert `Time` to datetime  
☑ Standardize `Direction`  
Check `Floor` column: Are all values integers? Any outliers (e.g., 0, 999)?  
Deduplication: Are there fully duplicated rows?

**car_calls.csv**  
Convert `Time` to datetime  
Clean `Floor` column (e.g., handle basement floors like B1 → -1)  
Check `Car ID` range (any IDs exceeding total elevator count?)

**load_changes.csv**  
Convert `Time` to datetime  
Is `Load (%)` within 0–100%? Any negative values or >100%?  
**Critical**: Sort by `Car ID` + `Time` to ensure continuous time series.

**maintenance_mode.csv**  
Convert `Time` to datetime  
`Status` column: Boolean (`True`/`False`) or string (`"Active"`/`"Maintenance"`)?  
**Must generate an "operational period" mask** for filtering other datasets!

**car_stops.csv**  
Convert `Time` to datetime  
Clean `Floor` (same as in `car_calls`)  
Verify if stops align with `hall_calls`/`car_calls` (optional consistency check)

Our file structure remains largely unchanged:
```
MCM2026 Training Test Problem B/
├── mcm26Train-B-Data/
│   ├── hall_calls.csv
│   ├── car_stops.csv
│   └── ... (other .csv and .md files as listed above)
├── data_cleaning/
│   ├── hall_calls_clean.csv          ← Done
│   ├── ..._clean.csv                 ← Not done
│   ├── ...
│   ├── hourly_calls.pdf              ← Done
│   ├── hourly_up_down.pdf            ← Done
│   ├── hall_calls_heatmap_weekday.pdf   ← Done
│   ├── hall_calls_heatmap_weekend.pdf   ← Done
│   ├── hourly_calls_weekday_vs_weekend.pdf ← Done
│   ├── response_time_distribution.pdf   ← Done
│   └── elevator_load_by_hour.pdf     ← Done
└── Untitled.ipynb                    ← Created Jupyter Notebook
```

Should we just go ahead and clean these files? It’s better than leaving them uncleaned, right?  
If yes, please first clearly specify the cleaning criteria for each file, then provide the code.


**Output:**
---
**Excellent decision!** Although your core task primarily relies on `hall_calls`, **performing basic cleaning on all datasets and saving them as `_clean.csv` files** will not only enhance project completeness but also provide strong support during later modeling (e.g., validation, feature engineering) or if reviewers raise questions.

Below, I’ll **specify cleaning standards per file + provide ready-to-run Python code**, ensuring:
- Only **necessary and safe** cleaning steps are applied (no original information destroyed)
- Output is clear, well-typed `_clean.csv` files
- All valid records are preserved (unless clearly erroneous)

---

## ✅ General Principles
1. **Do not delete plausible outliers** (e.g., Load=105% might be sensor noise—keep for now)
2. **No imputation or guessing** (drop or flag missing values)
3. **Standardize time format to `datetime64[ns]`**
4. **Output to `data_cleaning/` directory**

---

## 📁 1. `hall_calls.csv` (already cleaned – just verification)

### Cleaning Standards
- ✅ Convert `Time` to datetime  
- ✅ Standardize `Direction` to `'Up'` / `'Down'`  
- ✅ Ensure `Floor` is integer; drop NaNs  
- ❌ Do **not** remove floor outliers (e.g., 0, 999)—they might represent real floors (e.g., basement labeled as 0)

> ✅ Already done—no need to redo. But consider adding: **check if Floor range is reasonable (e.g., 1–20)**

```python
# Optional: Check Floor range
print("Hall Calls Floor range:", df_hall['Floor'].min(), "to", df_hall['Floor'].max())
# If you see 0 or >30, manually verify—but don’t delete yet
```

---

## 📁 2. `car_calls.csv` (in-car destination calls)

### Cleaning Standards
| Column | Action |
|--------|--------|
| `Time` | Convert to `datetime` |
| `Car ID` | Keep as-is (assume 1–8); check for unreasonable values (e.g., >10?) |
| `Floor` | **Critical**: Handle negative/basement labels (e.g., `"B1"` → `-1`, `"G"` → `1`); drop unparseable rows |

> 💡 Assumed floor naming convention:  
> - `B1`, `B2` → `-1`, `-2`  
> - `G`, `Lobby` → `1`  
> - Numeric strings → convert to int

### ✅ Cleaning Code

```python
import pandas as pd
import os
import re

# Load raw data
df_car = pd.read_csv('mcm26Train-B-Data/car_calls.csv')

# 1. Convert Time to datetime
df_car['Time'] = pd.to_datetime(df_car['Time'])

# 2. Standardize Floor column
def parse_floor(floor_str):
    if pd.isna(floor_str):
        return None
    s = str(floor_str).strip().upper()
    # Handle basements
    if s.startswith('B') and s[1:].isdigit():
        return -int(s[1:])
    # Handle ground floor
    if s in ['G', 'GROUND', 'LOBBY', '1']:
        return 1
    # Try converting to integer
    try:
        return int(float(s))  # handles '2.0'
    except:
        return None  # unparseable

df_car['Floor'] = df_car['Floor'].apply(parse_floor)

# 3. Drop rows with unparseable Floor
df_car_clean = df_car.dropna(subset=['Floor']).copy()
df_car_clean['Floor'] = df_car_clean['Floor'].astype(int)

# 4. (Optional) Check Car ID range
print("Car ID unique values:", sorted(df_car_clean['Car ID'].unique()))
# If anomalies found (e.g., Car ID=99), filter: df_car_clean = df_car_clean[df_car_clean['Car ID'] <= 8]

# 5. Save
os.makedirs('data_cleaning', exist_ok=True)
df_car_clean.to_csv('data_cleaning/car_calls_clean.csv', index=False)
print(f"✅ car_calls_clean.csv saved. Shape: {df_car_clean.shape}")
```

---

## 📁 3. `load_changes.csv` (elevator load changes)

### Cleaning Standards
| Column | Action |
|--------|--------|
| `Time` | Convert to datetime |
| `Car ID` | Keep as-is |
| `Load (%)` | Keep all values (even <0 or >100), but **flag anomalies**; do not delete |

> ⚠️ Load exceeding limits may be sensor noise, but still useful for peak detection

### ✅ Cleaning Code

```python
df_load = pd.read_csv('mcm26Train-B-Data/load_changes.csv')

# 1. Convert Time to datetime
df_load['Time'] = pd.to_datetime(df_load['Time'])

# 2. (Optional) Add anomaly flag
df_load['Load_Anomaly'] = (df_load['Load (%)'] < 0) | (df_load['Load (%)'] > 100)

# 3. Sort by Car ID + Time (critical!)
df_load_clean = df_load.sort_values(['Car ID', 'Time']).reset_index(drop=True)

# 4. Save
df_load_clean.to_csv('data_cleaning/load_changes_clean.csv', index=False)
print(f"✅ load_changes_clean.csv saved. Shape: {df_load_clean.shape}")
print("Load range:", df_load_clean['Load (%)'].min(), "to", df_load_clean['Load (%)'].max())
```

---

## 📁 4. `maintenance_mode.csv` (maintenance status)

### Cleaning Standards
| Column | Action |
|--------|--------|
| `Time` | Convert to datetime |
| `Status` | Standardize to boolean: `True` = operational, `False` = under maintenance |
| **Key** | Generating “operational periods” is for later—here, only clean |

> Assume original `Status` is string like `"Active"` / `"Maintenance"`

### ✅ Cleaning Code

```python
df_maint = pd.read_csv('mcm26Train-B-Data/maintenance_mode.csv')

# 1. Convert Time to datetime
df_maint['Time'] = pd.to_datetime(df_maint['Time'])

# 2. Standardize Status to boolean
def parse_status(s):
    if pd.isna(s):
        return False  # default to unavailable
    s = str(s).strip().lower()
    return s in ['active', 'true', '1', 'running']

df_maint['Is_Operational'] = df_maint['Status'].apply(parse_status)

# 3. Save (keep original Status + add boolean column)
df_maint_clean = df_maint[['Time', 'Car ID', 'Status', 'Is_Operational']].copy()

df_maint_clean.to_csv('data_cleaning/maintenance_mode_clean.csv', index=False)
print(f"✅ maintenance_mode_clean.csv saved. Operational ratio: {df_maint_clean['Is_Operational'].mean():.2%}")
```

---

## 📁 5. `car_stops.csv` (elevator stop records)

### Cleaning Standards
- Same `Floor` logic as `car_calls.csv`  
- Convert `Time` to datetime  
- Keep all valid stops

### ✅ Cleaning Code

```python
df_stops = pd.read_csv('mcm26Train-B-Data/car_stops.csv')

# 1. Convert Time to datetime
df_stops['Time'] = pd.to_datetime(df_stops['Time'])

# 2. Reuse parse_floor function (redefine here)
def parse_floor(floor_str):
    if pd.isna(floor_str):
        return None
    s = str(floor_str).strip().upper()
    if s.startswith('B') and s[1:].isdigit():
        return -int(s[1:])
    if s in ['G', 'GROUND', 'LOBBY', '1']:
        return 1
    try:
        return int(float(s))
    except:
        return None

df_stops['Floor'] = df_stops['Floor'].apply(parse_floor)
df_stops_clean = df_stops.dropna(subset=['Floor']).copy()
df_stops_clean['Floor'] = df_stops_clean['Floor'].astype(int)

# 3. Save
df_stops_clean.to_csv('data_cleaning/car_stops_clean.csv', index=False)
print(f"✅ car_stops_clean.csv saved. Shape: {df_stops_clean.shape}")
```

---

## 📁 Final File Structure

```
MCM2026 Training Test Problem B/
├── data_cleaning/
│   ├── hall_calls_clean.csv
│   ├── car_calls_clean.csv
│   ├── load_changes_clean.csv
│   ├── maintenance_mode_clean.csv
│   ├── car_stops_clean.csv
│   └── *.pdf (your plots)
```

---

## 📝 How to Mention in Your Paper?

Add one sentence in the Data Preprocessing section:

> “All raw datasets underwent basic cleaning: timestamp standardization, categorical encoding, and removal of unparseable records. Cleaned versions are archived in the `data_cleaning/` directory for reproducibility.”

---

You can now **run the code blocks above sequentially** (one cell per file) to complete all cleaning.  
<<<<<<< HEAD
Would you like me to combine all this into a single, ready-to-run notebook script?
=======
Would you like me to combine all this into a single, ready-to-run notebook script?
>>>>>>> fb559f0c2080e58b8b3f046a4a92650848596a70
