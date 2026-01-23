Memo: "Night Idle" Threshold (theta1) Computed as Zero — Issue, Impact, and Recommended Fixes

To: Modeling Team
From: Programming Lead (on behalf of modeling team)
Date: 2026-01-22

Summary

During implementation and testing of the Task 2 rule-based traffic-state classifier, we observed that the low-activity threshold `theta1` — computed as the 10th percentile of 5-minute total call counts — evaluates to 0 for our cleaned dataset. Because the classifier uses the rule

  if C < theta1: return 'Night Idle'

this means no interval can satisfy C < 0 and the `Night Idle` label never occurs. This memo explains why this happened, what the practical impact is, and proposes safe, reversible fixes plus an action plan for the group to adopt.

Background and why it happened

- Our current `compute_thresholds()` uses the dataset-wide quantile of the raw per-interval `count` distribution to produce `theta1` (default quantile: 0.10).
- In this building’s 5-minute aggregation, many intervals have zero calls (quiet night hours or weekends). The 10th percentile of the full distribution therefore equals 0, which is statistically correct but semantically unhelpful for the `Night Idle` rule.
- The classification decision was written as a strict inequality `C < theta1`, which cannot be true when theta1 == 0 (C is count >= 0). Even if `C <= theta1` were used, only zero-count intervals would be labelled Night Idle, but the intent is to capture low-activity periods, not only absolute zeros.

Practical impact

- `Night Idle` is effectively disabled. Intervals that a human operator would regard as “night idle” (for example, 0–2 calls per 5 minutes) are instead being assigned other states, typically `Weekend Low-Demand` or `Afternoon Mixed` depending on the logic order.
- Downstream tasks relying on the state label (Task 3 parking logic, state‑aware evaluation) may behave suboptimally because they won’t see an explicit Night Idle mode for policy switches (e.g., aggressive energy saving or minimal lobby staffing)
- Metrics by state become less interpretable: the `Weekend Low-Demand` or `Afternoon Mixed` aggregates now include many very-low-activity intervals that dilute their meaning.

Recommended fixes (safe and incremental)

I list several options in order of minimal invasiveness to more robust structural changes. Each is reversible and testable.

1) Quick fix — set a small positive minimum for `theta1` (recommended default: 1)
   - Change the computed `theta1` to: max(quantile_value, 1)
   - Rationale: treats any interval with 0 calls as obvious idle; intervals with 1 call in 5 minutes are still extremely low and will be labelled `Night Idle` if desired depending on `C < theta1` vs `C <= theta1` rule refinement.
   - Pros: minimal code change, immediate behavioral fix, easy to revert.
   - Cons: hard-coded constant may not generalize to buildings with lower/higher base rates.

2) Pragmatic percentile-of-nonzero approach
   - Compute `theta1` as the q-th percentile (e.g., 10th) of the distribution of non-zero counts only, or use the 20th percentile of all counts.
   - Rationale: ignores the mass at zero introduced by night/weekend inactivity and finds a meaningful low-activity cutoff among times when activity occurs.
   - Pros: adaptive per dataset (still driven by quantiles), no magic constants.
   - Cons: if the dataset has long quiet periods, the nonzero-only percentile may push theta1 above what we’d want to call “idle.”

3) Time-aware thresholding (recommended for production)
   - Compute separate `theta1` thresholds per coarse time-window (e.g., night vs day) or use a time-of-day baseline: e.g., for `tau` in night hours (00:00–06:00) set `theta1_night` = X, elsewhere use different rule.
   - Rationale: the meaning of “idle” is context-dependent; 1 call in a 5-minute slot at 02:00 is different from 1 call at 12:00.
   - Pros: aligns with operational semantics; robust across buildings and seasonality.
   - Cons: more code and parameter choices; requires agreeing on time-windows.

4) Rule rewrite: use <= and explicit low-activity band
   - Replace `if C < theta1` with `if C <= theta1` and/or use a band: e.g., `if C <= theta_low: Night Idle; elif C < theta_medium: Quiet/Low-Demand`.
   - Rationale: makes the rule inclusive and clearer.
   - Pros: simple and interpretable.
   - Cons: still depends on reasonable theta values.

5) Hybrid: compute `theta1` as the lesser of (a small absolute constant, e.g., 1) and (10th percentile of nonzero counts) or similar heuristics.
   - This protects against extreme datasets while keeping adaptivity.

Suggested default for this project (one-line recommendation)

- Implement option (2) as primary: compute `theta1` from the 10th percentile of non-zero counts, and fall back to `theta1 = 1` if there are too few non-zero samples. Also change the condition to `C <= theta1` for inclusivity. This is adaptive, minimal, and interpretable.

Concrete code change (where to modify)

- File: `src/task2_classifier.py`
- Function: `compute_thresholds(features)`
- Replace the current `theta1` calculation with something like:

  nonzero = features['count'][features['count'] > 0]
  if len(nonzero) >= 30:
      theta1 = float(nonzero.quantile(0.10))
  else:
      theta1 = 1.0
  theta1 = max(1.0, theta1)  # enforce at least 1

- And in `classify_interval`, change the test to `if C <= theta1: return 'Night Idle'` (or keep `<` if you prefer exclusive semantics).

Tests and validation

- I will implement the change and re-run the classification demo and the integrated demo to produce:
  - New state counts (verify `Night Idle` appears and low-activity intervals shift appropriately)
  - Per-state summary metrics (ensure no unexpected deterioration in forecasting metrics)
  - A small sanity report with examples of intervals newly classified as `Night Idle`.

Action items & timeline

- If you approve the recommended default (option 2 + `C <= theta1`), I will update `src/task2_classifier.py`, run the integrated demo, and commit the small change. Estimated time: ~10–20 minutes.
- If you prefer a different approach (e.g., option 3 time-aware thresholds), tell me which time windows you want for “night” vs “day” (default night: 00:00–06:00). Estimated time: ~30–60 minutes to implement and test.

Decision requested

Please confirm one of the following so I can proceed automatically:
- Approve the recommended change: compute `theta1` from the 10th percentile of non-zero counts and use `C <= theta1` (default fallback theta1 = 1). (recommended)
- Approve the quick fix: force `theta1 = max(quantile, 1.0)` and change to `C <= theta1`.
- Request the time-aware approach and provide preferred night time window (default 00:00–06:00 will be used if you don't answer).
- Ask me to implement a different rule (describe it).

If I don't hear back in this session I will not change code automatically; once you confirm I will make the code change, re-run the integrated demo, and report the updated metrics and sample classifications.

Appendix: quick examples

- Current behavior: when theta1 == 0, no interval is classified as `Night Idle`.
- With recommended change: a 5-min interval with `count` = 0 or 1 during the night will typically be labelled `Night Idle` (depending on the percentile), restoring the intended semantic meaning.

End of memo.
