TL;DR — Build reproducible pipelines that implement the report’s three components: (1) 5‑minute short‑term passenger flow predictor (time‑of‑day baseline + regime AR(1)), (2) rule‑based real‑time traffic state classifier, and (3) state‑aware dynamic elevator parking policy; validate with the cleaned datasets and provide scripts/notebook for backtesting and tuning.

Steps
1. Inspect cleaned data: review `data_cleaning/hall_calls_clean.csv` and `data_cleaning/readme.md`.
2. Prepare data pipeline: create loader/feature functions for 5‑minute bins, `Hour`, `Weekday/Is_Weekend`, direction and floor‑aggregates.
3. Implement Task 1: compute `mu_w(tau)` (time‑of‑day baseline) and fit separate AR(1) phi for weekday/weekend; expose `predict_next(y_t, t)` in a module.
4. Implement Task 2: extract real‑time features (`C_t`, `C_up`, `C_down`, `r_t`, `p1_t`, `H_t`) and implement rule logic from the report producing state `s_t`.
5. Implement Task 3: encode zone set `L={Lobby,Mid,Upper}`, map states + `y_hat` to parking assignments and reposition triggers.
6. Integration & evaluation: backtest on historical data (simulate 5‑min steps), log AWT proxy, long‑wait events, and empty travel; produce metrics and plots in a notebook.
7. Deliverables: `src/` modules for Tasks 1–3, `notebooks/` for EDA and backtest, `requirements.txt`, and `README.md` with run instructions.

Further Considerations
1. Model complexity choice: Option A — keep AR(1)+baseline (report‑recommended, interpretable); Option B — upgrade to LSTM/GBM if accuracy needed. Which do you prefer?
2. Evaluation metrics and simulation: define AWT proxy and empty‑travel metric; determine threshold τ for long‑wait events before backtesting. Recommend starting with τ = 60s.
3. Operational details: confirm building specifics (elevator count, lobby floor index) and decide whether to simulate elevator motion or use a simplified distance/probability proxy.

This is a draft plan — tell me which options you prefer (A/B for model complexity; τ value; simulation fidelity) or any constraints (time, preferred languages, CI/tests) and I’ll refine the next plan iteration.
