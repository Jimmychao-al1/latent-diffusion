# Stage2 (LDM): Runtime Refine

## Purpose
Stage2 reads Stage1 `scheduler_config.json`, runs baseline/cache diagnostics, then performs a single conservative refinement pass:

1. Zone-level adjust: if zone mean L1 is too high, decrease `k_per_zone` by 1 (min 1).
2. Peak-level repair: if per-step L1 is too high, force `expanded_mask[step_idx]=True`.
3. Enforce first step full compute: `expanded_mask[0]=True`.

## LDM-specific assumptions
- `T=200`
- `time_order=ddim_descending_Tminus1_to_0`
- runtime block count = `25` (`encoder_layer_0..11`, `middle_layer`, `decoder_layer_0..11`)

## Files
- `stage2_scheduler_adapter_ldm.py`
- `stage2_error_collector_ldm.py`
- `stage2_runtime_refine_ldm.py`
- `build_blockwise_thresholds_ldm.py`
- `verify_stage2_ldm.py`
- `run_stage2_full_experiments_ldm.sh`
- `export_stage2_diagnostics_csv_ldm.py`

## Default threshold policy
`build_blockwise_thresholds_ldm.py` defaults are ported from baseline_908030:
- `q_zone=0.90`
- `q_peak=0.80`
- `peak_over_zone_ratio_min=1.3`
- metadata `source=\"ported_from_baseline_908030\"`

## Full run (3 sources, Pass1+Pass2 each)
From repo root:

```bash
bash ldm_S3cache/cache_method/Stage2/run_stage2_full_experiments_ldm.sh
```

Defaults in this script:
- `eval_num_images=8`
- `eval_chunk_size=1`
- sources:
  - `sweep_K15_sw3_lam1.0`
  - `sweep_K15_sw3_lam0.5`
  - `sweep_K20_sw3_lam1.0`

## Per-source output layout
Root:
- `ldm_S3cache/cache_method/Stage2/stage2_output_ldm/<src_name>/`

Inside each source:
- `00_global_refine/stage2_runtime_diagnostics.json`
- `00_global_refine/stage2_refined_scheduler_config.json`
- `00_global_refine/stage2_refinement_summary.json`
- `00_global_refine/cache_runtime_overrides_run.json`
- `01_blockwise_threshold/stage2_thresholds_blockwise.json`
- `02_refined_blockwise/stage2_runtime_diagnostics.json`
- `02_refined_blockwise/stage2_refined_scheduler_config.json`
- `02_refined_blockwise/stage2_refinement_summary.json`
- `02_refined_blockwise/cache_runtime_overrides_run.json`

## Manual commands
Pass1 (global):

```bash
python ldm_S3cache/cache_method/Stage2/stage2_runtime_refine_ldm.py \
  --scheduler_config <stage1_scheduler_config.json> \
  --output_dir <out>/00_global_refine \
  --seed 0 \
  --zone_l1_threshold 0.02 \
  --peak_l1_threshold 0.08 \
  --eval-num-images 8 \
  --eval-chunk-size 1
```

Build blockwise thresholds:

```bash
python ldm_S3cache/cache_method/Stage2/build_blockwise_thresholds_ldm.py \
  --diagnostics <out>/00_global_refine/stage2_runtime_diagnostics.json \
  --output <out>/01_blockwise_threshold/stage2_thresholds_blockwise.json
```

Pass2 (blockwise):

```bash
python ldm_S3cache/cache_method/Stage2/stage2_runtime_refine_ldm.py \
  --scheduler_config <stage1_scheduler_config.json> \
  --output_dir <out>/02_refined_blockwise \
  --seed 0 \
  --zone_l1_threshold 0.02 \
  --peak_l1_threshold 0.08 \
  --threshold-config <out>/01_blockwise_threshold/stage2_thresholds_blockwise.json \
  --eval-num-images 8 \
  --eval-chunk-size 1
```

Verify outputs:

```bash
python ldm_S3cache/cache_method/Stage2/verify_stage2_ldm.py \
  <out>/02_refined_blockwise/stage2_refined_scheduler_config.json

python ldm_S3cache/cache_method/Stage2/verify_stage2_ldm.py \
  --threshold-config <out>/01_blockwise_threshold/stage2_thresholds_blockwise.json
```

Export diagnostics to one consolidated CSV:

```bash
python ldm_S3cache/cache_method/Stage2/export_stage2_diagnostics_csv_ldm.py
```

Default output:
- `ldm_S3cache/cache_method/Stage2/stage2_output_ldm/csv_exports/stage2_runtime_diagnostics_combined.csv`
