# Problem (benchmarking_script) Responses (Draft)

## (b) 5 warmup, 10 measurement steps
- Fill numbers from results_b_summary.csv.
- 1-2 sentence draft:
  Forward passes are on the order of <fill> s across model sizes, while backward passes are on the order of <fill> s and are consistently slower than forward. The standard deviations are <fill small/large>, indicating <fill variability observation>.

## (c) Warmup ablation (0/1/2/5)
- Fill numbers from results_c_summary.csv.
- 2-3 sentence draft:
  Without warm-up, measured times are typically higher and more variable than with 5 warm-up steps, because initial iterations include one-time overheads (e.g., CUDA context setup, allocator behavior, and kernel/runtime initialization). Using only 1-2 warm-up steps reduces this effect but does not always eliminate it, since not all execution paths and caches are fully stabilized yet. As warm-up increases, timings generally become more stable and representative of steady-state throughput.

## Quick Reference Tables

### Problem (b)
| size | mode | status | device | dtype | warmup_steps | measure_steps | mean_seconds | std_seconds | num_parameters | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | backward | ok | cpu | torch.float32 | 5 | 1 | 0.4219649860001482 | nan | 128625408 |  |
| small | forward | ok | cpu | torch.float32 | 5 | 1 | 0.23871572500002003 | nan | 128625408 |  |

### Problem (c)
| size | mode | status | device | dtype | warmup_steps | measure_steps | mean_seconds | std_seconds | num_parameters | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small | backward | ok | cpu | torch.float32 | 0 | 1 | 0.442300097000043 | nan | 128625408 |  |
| small | forward | ok | cpu | torch.float32 | 0 | 1 | 0.22118863400010014 | nan | 128625408 |  |
