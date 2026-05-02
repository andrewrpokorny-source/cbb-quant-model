# Market V2 Production-Readiness Diagnostics

Config: `mlb_research/anchor_v2/configs/lgbm_top13_market.json`
Anchor sha256: `c7613525d02009d3bc4807954a6aa502907ce17039b2c48869b2aa4993e34996`

| Window | N | Brier Delta | Log Loss Delta | ROI Delta | Model ROI | Market ROI | Pick Agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| optimizer | 1378 | -0.002839 | -0.005556 | 126.37 | 51.47 | -74.90 | 0.657 |
| monitor_2025_tail | 727 | -0.005951 | -0.012161 | 87.54 | 49.17 | -38.36 | 0.708 |
| monitor_2026 | 83 | -0.055843 | -0.117955 | 38.51 | 36.78 | -1.73 | 0.627 |

## Bootstrap 95% Intervals

- `optimizer`: Brier delta [-0.008003, 0.002513], log loss delta [-0.016695, 0.006057], ROI delta [57.73, 189.92]
- `monitor_2025_tail`: Brier delta [-0.011532, -0.000631], log loss delta [-0.023636, -0.001259], ROI delta [43.33, 133.11]
- `monitor_2026`: Brier delta [-0.085084, -0.022822], log loss delta [-0.179439, -0.047707], ROI delta [17.68, 60.89]

## Production Read

The candidate clears the first production-readiness screen: paired Brier is better than market-only in every frozen window and monitor ROI remains positive. Next step is prediction-time odds plumbing, with this model still behind a comparison/reporting flag.
Optimizer ROI should not be treated as a promotion criterion by itself; the paired optimizer Brier delta is `-0.002839`.

## Watchouts

- `optimizer` paired Brier bootstrap interval crosses zero: [-0.008003, 0.002513].
- Some month-level splits are worse than market-only: `optimizer` 2025-04 n=263 Brier delta=0.006048.
- Optimizer ROI edge is concentrated in model-selected dogs: dog ROI delta `105.79U`, favorite ROI delta `20.58U`.
- Monitor 2026 is still a small sample (`n=83`); use it as a smoke check, not a promotion proof.
