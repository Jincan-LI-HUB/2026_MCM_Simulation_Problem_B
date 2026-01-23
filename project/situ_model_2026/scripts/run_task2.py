#!/usr/bin/env python3
import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.task2_classifier import build_interval_features, compute_thresholds, classify_all


def main():
    csv_path = os.path.join(repo_root, 'data_cleaning', 'hall_calls_clean.csv')
    features = build_interval_features(csv_path)
    thr = compute_thresholds(features)
    print('Thresholds:', thr)
    states = classify_all(features, thr)
    print('\nState counts:')
    print(states.value_counts())
    print('\nSample:')
    print(states.head(20))


if __name__ == '__main__':
    main()
