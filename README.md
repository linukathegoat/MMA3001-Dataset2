# Source measurements

Dataset: `5EnvSensor_MayToDec2024_180kRows.zip` provided by the MMA3001 teaching team (Monash Smart Infrastructure environmental sensors).

Input schema: zipped CSV fields `sensorid` and `jsondata`. Each JSON temperature item has a sensor-specific value and measurement timestamp (Unix seconds). The outer record timestamp is not used as the measurement time. Measurements are filtered to May 2024 UTC for sensor `6012002000869`.

The source archive is not distributed with this repository. Code reads the original ZIP directly and does not require the 9-million-row occupancy file. The original data and the locally exported cleaned time series must remain outside the public repository unless redistributable rights are confirmed.
