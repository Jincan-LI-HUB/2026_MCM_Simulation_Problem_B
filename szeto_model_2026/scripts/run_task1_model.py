#!/usr/bin/env python3
import os
from src.task1_model import build_5min_series, estimate_baseline_and_phi, predict_next

CSV_PATH = os.path.abspath(r"C:\Users\admin\PycharmProjects\model2026\data_cleaning\hall_calls_clean.csv")


def main():
    series = build_5min_series(CSV_PATH)
    mu_w, phi_w = estimate_baseline_and_phi(series)
    print('Estimated phi_w:', phi_w)
    print('Weekday mu (0:60 min):', mu_w[1][:12])
    print('Weekend mu (0:60 min):', mu_w[0][:12])
    # sample prediction for the last observed interval
    last_ts = series.index[-1]
    last_count = int(series['count'].iloc[-1])
    next_ts, yhat = predict_next(last_ts, last_count, mu_w, phi_w)
    print(f'Sample next interval {next_ts} predicted count: {yhat:.3f} (last_count={last_count})')


if __name__ == '__main__':
    main()
