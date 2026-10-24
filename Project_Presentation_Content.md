# Missing Temperature Data Reconstruction

MMA3001 — Numerical Methods and Machine Learning

## 1. Engineering problem
Building monitoring systems record indoor temperatures over time. Short communication or logging failures can create missing samples. The numerical task is to reconstruct an interior missing interval from surrounding genuine observations, rather than forecast a future temperature.

## 2. Dataset and preprocessing
Monash environmental sensor `6012002000869`, Building 60 / G.38 (EMS107), May 2024 UTC. Temperature values and their actual timestamps were extracted from the nested JSON field. Four same-timestamp records were consolidated, leaving 4,155 unique temperature readings. This avoids the unreliable cross-mapping between occupancy and environmental sensors.

## 3. Computational methods
Last-observation-carried-forward provides a simple baseline. Linear interpolation connects two measured endpoints with a straight line. PCHIP preserves local shape using piecewise cubic polynomials. Natural cubic splines impose smooth derivative continuity. Only interpolation between observed endpoints is allowed.

## 4. Validation and comparison
A chronological 70/30 partition separates method selection from later evaluation. Fifteen disjoint missing blocks of 3, 6 and 12 samples were simulated in validation. Measured values withheld from the algorithm provided evaluation targets. Linear interpolation achieved the lowest mean validation MAE, including 0.116 °C for six-reading gaps.

On the later holdout, linear interpolation had 0.147 °C MAE and PCHIP 0.131 °C MAE. Model selection based on an earlier period did not guarantee the smallest error in the later period.

## 5. Interpretation, performance, verification and limitations
Increasing missing-gap length generally increased errors. Linear interpolation offered strong accuracy and low computational overhead. Ten automated software tests passed. However, simulated missingness, use of one sensor and one month, imperfect sensor calibration and correlated readings limit generalisation. The method is designed for retrospective data repair, not real-time safety-critical decisions.

## AI contribution
ChatGPT materially assisted with project design, Python implementation, tests, reporting and explanation. Computational tests, engineering validation and independent critical review remain separate requirements.
