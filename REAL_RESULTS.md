# Actual numerical results — MMA3001 Dataset 2

Sensor: `6012002000869` — Monash Building 60, G.38, Compactus_upper (label EMS107).

## Data audit
- Study interval: 2024-05-01 to 2024-06-01 (UTC, exclusive end)
- Original messages from this sensor in entire export: 29,341
- Temperature values in May study interval: 4,159
- Distinct May readings: **4,155**
- Repeated same-timestamp messages removed: 4
- Conflicting same-timestamp readings: 0
- Median original sampling interval: 10.00 min
- Observed temperature range: 19.0 to 25.5 °C

## Validation (chronologically earlier 70% of May data)

Masked known readings, then predicted without training on those hidden targets.
The following table gives validation MAE in °C. Blocks of 3, 6 and 12 readings
were tested separately using 15 non-overlapping blocks per gap size.

| method       |      3 |      6 |     12 |
|:-------------|-------:|-------:|-------:|
| cubic_spline | 0.1227 | 0.2204 | 0.2799 |
| last_known   | 0.1356 | 0.2378 | 0.7128 |
| linear       | 0.0969 | 0.1159 | 0.2243 |
| pchip        | 0.0986 | 0.1339 | 0.2052 |

**Selected method, based only on validation MAE: `linear`.**

## Independent later-May holdout (last 30%, six-reading gaps)

| method       |   n_hidden_readings |   mean_gap_duration_minutes |   MAE_C |   RMSE_C |   MaxAE_C |   within_0_5_C_pct |   median_prediction_runtime_ms |
|:-------------|--------------------:|----------------------------:|--------:|---------:|----------:|-------------------:|-------------------------------:|
| last_known   |                  90 |                     76.0033 |  0.2700 |   0.4802 |    2.0000 |            93.3333 |                         0.0268 |
| linear       |                  90 |                     76.0033 |  0.1467 |   0.2648 |    1.2373 |            94.4444 |                         0.0203 |
| pchip        |                  90 |                     76.0033 |  0.1307 |   0.2095 |    0.9178 |            95.5556 |                         0.2157 |
| cubic_spline |                  90 |                     76.0033 |  0.1597 |   0.2123 |    0.6995 |            96.6667 |                         0.1688 |

## Important interpretation limits

- All withheld values were real measured readings; the missingness was simulated, not naturally missing ground truth.
- Testing only one sensor and month does not demonstrate performance elsewhere.
- No method may extrapolate past the first/last retained observation.
- Adjacent readings are correlated, so hidden predictions are not fully independent.
- Temperature values could contain sensor bias or calibration errors; no physical ground-truth instrument was available.
- Timing uses a local process and includes estimator fitting; it will vary by computer.
- Source CSV filename references May–December 2024 but individual JSON times also include other years; this project explicitly filters **May 2024 timestamps**.

## Reproducibility

Run `python run_project.py --input <environment-zip-path>` from project root.
Check generated CSV files, PNGs and `pytest -q`. Do not upload raw proprietary or private sensor logs to a public repository without permission.
