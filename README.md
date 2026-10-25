# Reconstruction of Missing Indoor Temperature Data

**MMA3001 — Numerical Methods and Machine Learning | Individual computational project**

## Research objective

The project evaluates four numerical approaches to reconstructing missing temperature records in a building-monitoring time series. Known measured values are withheld in controlled blocks; the numerical estimates are compared against the withheld measurements. The method selected by chronological validation is evaluated again on a later independent holdout period.

**Research question:** How do interpolation method and missing-gap duration affect temperature reconstruction error and computational cost?

## Engineering system and dataset

The environmental sensor export supplied as MMA3001 Dataset 2 contains nested JSON observations. The study uses sensor `6012002000869` (Building 60, room G.38, Compactus_upper, EMS107, according to the unit-provided location workbook) over **1–31 May 2024, UTC**. Analysis uses the timestamps attached to individual JSON readings rather than the outer database record timestamps. This scope does not assume any reliable pairing with occupancy sensors.

Four thousand one hundred fifty-five unique readings (19.0–25.5 °C) were retained. Four repeated measurements were consolidated; median sampling interval was 10 minutes. See `results/data_audit.json` for all recorded integrity checks. The original environmental archive is not redistributed.

## Numerical methods

| Name | Model | Main consideration |
| --- | --- | --- |
| `last_known` | Previous known temperature, held constant | Simple baseline; cannot represent within-gap changes |
| `linear` | Piecewise linear interpolation in elapsed time | Efficient and interpretable |
| `pchip` | Shape-preserving piecewise cubic Hermite interpolation | Controls artificial overshoot |
| `cubic_spline` | Natural cubic spline interpolation | Smooth derivatives but may overshoot |

Estimates are only calculated between known observations. Querying outside the measured domain raises an error. Duplicate readings are consolidated before interpolation, and the timestamp origin is shifted for numerical conditioning.

## Experimental design

Data are divided chronologically into an earlier 70% validation partition (2,908 observations) and a later 30% holdout partition (1,247 observations). Fifteen nonoverlapping blocks of 3, 6 and 12 known observations are hidden within the validation partition. Candidate gaps containing original sampling steps over 30 minutes are excluded. Interpolators are fitted to unhidden readings; no hidden target is used as an interpolator input.

Methods are ranked by validation mean absolute error (MAE), averaged equally across the three missing-gap lengths. The winner is then assessed on 15 new six-observation gaps from the later holdout. Reported metrics include MAE, RMSE, maximum absolute error, percentage within 0.5 °C and runtime of repeated prediction calls. Gap-length sensitivity and comparative computational performance are evaluated.

## Main findings

**Linear interpolation** achieved the smallest average validation MAE. On the six-reading validation gaps, its MAE was **0.116 °C**, compared with **0.238 °C** for the last-observation baseline, **0.134 °C** for PCHIP and **0.220 °C** for the natural cubic spline. For 12-reading gaps, PCHIP was slightly better than linear (0.205 °C versus 0.224 °C MAE).

On later holdout gaps, the preselected linear method obtained **0.147 °C MAE** and **0.265 °C RMSE**. PCHIP gave the lowest holdout MAE (**0.131 °C**). This difference limits the strength of claims about the universally preferable method. Source tables and plots are stored in `results/` and `figures/`.

## Reproducibility

### Google Colab notebook

`MMA3001_Temperature_Reconstruction.ipynb` contains the complete source code used for the study, imports the environmental archive through Colab's upload interface, performs the analysis, runs automated tests, creates HTML documentation and displays the figures. The notebook is independently runnable with the original environmental archive and does not require a separate project ZIP.

### Python command line

```bash
python -m pip install -r requirements.txt
python run_project.py --input /path/to/5EnvSensor_MayToDec2024_180kRows.zip
python -m pytest -q
```

The repository uses Python, NumPy, pandas, SciPy, Matplotlib, pytest and tabulate. Charts and aggregate metrics are generated under `figures/` and `results/`. Environment-dependent timing values may vary between runs.

## Repository structure

| Location | Contents |
| --- | --- |
| `MMA3001_Temperature_Reconstruction.ipynb` | Google Colab experimental notebook |
| `run_project.py` | Numerical analysis, metric exports and figure generation |
| `src/sensor_project.py` | Data parsing, algorithms, masking and validation functions |
| `tests/test_sensor_project.py` | Ten automated software tests |
| `docs/` | HTML documentation generated from Python docstrings |
| `results/` | Computed aggregate tables, integrity audit and test record |
| `figures/` | Six numerical analysis figures |
| `report/` | Written project documentation (Word and PDF) |
| `data/README.md` | Source-data description and access restrictions |

## Validation scope and limitations

The target values used for validation are original sensor readings, while missingness is simulated. This does not guarantee performance during actual telemetry interruptions. Data are limited to one month, one sensor and one measurement type, and adjacent values are correlated. No calibrated reference thermometer was available to quantify systematic measurement bias. The approach is a **retrospective reconstruction method**, not a future temperature forecast or a safety-critical control method.

## AI-use disclosure

OpenAI ChatGPT provided substantial assistance to debug and modify Python code, identify and fix errors, and to improve the technical report. However, all computational predictions and tests are generated from the source code and the supplied sensor archive. AI assistance is documented in the repository and written report.

## Source and licence

The sensor archive and mapping workbook are coursework materials supplied by Monash University. The underlying data are not covered by this repository's software licence and are not redistributed. The accompanying MIT licence covers the code and project-authored documentation only, subject to the rights of contributors.
