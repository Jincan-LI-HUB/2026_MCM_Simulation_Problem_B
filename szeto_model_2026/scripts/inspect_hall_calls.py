#!/usr/bin/env python3
import os
import sys
import pandas as pd

CSV_PATH = os.path.abspath(r"C:\Users\admin\PycharmProjects\model2026\data_cleaning\hall_calls_clean.csv")
OUT_PATH = os.path.abspath(r"C:\Users\admin\PycharmProjects\model2026\scripts\hall_calls_inspect.txt")

def find_time_column(cols):
    lower = [c.lower() for c in cols]
    for candidate in ['time','timestamp','datetime','date','ts']:
        if candidate in lower:
            return cols[lower.index(candidate)]
    # try fuzzy
    for c in cols:
        if 'time' in c.lower() or 'date' in c.lower():
            return c
    return None


def main():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        sys.exit(2)

    df = pd.read_csv(CSV_PATH)
    cols = df.columns.tolist()

    lines = []
    lines.append(f"File: {CSV_PATH}")
    lines.append(f"Rows: {len(df):,}")
    lines.append(f"Columns ({len(cols)}): {cols}")
    lines.append('\nColumn dtypes:')
    lines.append(str(df.dtypes))
    lines.append('\nNull counts per column:')
    lines.append(str(df.isnull().sum()))

    time_col = find_time_column(cols)
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            lines.append(f"\nDetected time column: {time_col}")
            lines.append(f"Time range: {df[time_col].min()} --> {df[time_col].max()}")
            lines.append(f"Sample times (head): {df[time_col].head().tolist()}")
        except Exception as e:
            lines.append(f"\nDetected time column {time_col} but failed to parse as datetime: {e}")
    else:
        lines.append('\nNo obvious time column found.')

    for col in ['Direction','direction','Dir','dir']:
        if col in cols:
            vc = df[col].fillna('<<NA>>').value_counts().head(20)
            lines.append(f"\nValue counts for {col}:")
            lines.append(str(vc))
            break

    # Floor-like
    floor_col = None
    for col in cols:
        if 'floor' in col.lower():
            floor_col = col
            break
    if floor_col:
        lines.append(f"\nDetected floor column: {floor_col}")
        try:
            floors = pd.to_numeric(df[floor_col], errors='coerce')
            lines.append(f"Unique floors (sorted): {sorted(floors.dropna().unique())[:50]}")
            lines.append(f"Floor nulls: {floors.isnull().sum()} of {len(floors)}")
        except Exception as e:
            lines.append(f"Failed to process floor column: {e}")
    else:
        lines.append('\nNo floor column detected.')

    # Weekday/Is_Weekend
    for col in ['Weekday','weekday','Is_Weekend','is_weekend','isWeekend','weekday_name']:
        if col in cols:
            lines.append(f"\nDetected column: {col} -- value counts:\n{df[col].value_counts().head(20)}")
            break

    lines.append('\nSample rows:')
    lines.append(df.head(10).to_csv(index=False))

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print(f"\nWrote inspection output to: {OUT_PATH}")

if __name__ == '__main__':
    main()
