# windeval progress notes

Held-out evaluation of `scrollprize/winding_model_9um`. Newest entries at the bottom.

## 2026-10-08, day 1: setup

### What was downloaded

| Item | Where it came from | Local path | Size |
|---|---|---|---|
| `ckpt_final.pth`, `config.json`, `metrics.jsonl`, `.gitattributes` | HF `scrollprize/winding_model_9um`, revision `6e153b0eb51d1637185949750adffe7fa3f4f13e` (last modified 2026-08-14) | `~/kaggle-data/vesuvius/hf/winding_model_9um/` | 146,441,803 bytes for the checkpoint |
| PHerc0139 segment meshes, frame of the 2.399 um Jan 2026 scan (`mesh/<id>-on-20260102150214-2.399um.tifxyz`) | open-data bucket, `PHerc0139/segments/` | `~/kaggle-data/vesuvius/data/PHerc0139/segments/` | 935.8 MB, 38 segments |
| PHerc0139 meshes in the 9.362 um frame (`-on-20250728140407-9.362um.tifxyz`) | same | same | 61.9 MB, 38 segments |
| PHerc0139 `intermediate/tifxyz_original` and `tifxyz_normalized` | same | same | 105 MB |

Checkpoint sha256: `805a868c5ef518e4edb567ac2ddc63dfbf870f21b1164ae1642baa1781ad8f02`. It holds `model`, `model_ema` (9,135,724 fp32 parameters each), `optimizer`, `lr_scheduler`, `config` (identical to the HF `config.json`) and `step` = 50000. villa's inference script loads `model`, not `model_ema`, so `model` is the primary weight set here.

villa checkout: `e0bbb8b40a2db58b1d71864f286eb85717e59e64` (2026-10-06), untouched.

### Environment

Project venv `.venv` (uv, CPython 3.12.13), torch 2.14.1 with MPS. villa's `vesuvius/pyproject.toml` asks for Python 3.14; nothing in the winding-model files needed 3.14. Packages were installed one by one with `uv pip install` (no villa bootstrap scripts). An unrelated PyPI package called `vc` got pulled in by mistake during dependency probing and was uninstalled right away.

### The `vc` problem and how it was solved

The dataset code imports `vc`, villa's nanobind binding of volume-cartographer's C++ core. It does not build on macOS without open PR #1975, and it needs most of the VC3D dependency stack. The dataset uses only two pieces of it:

1. `vc.grid_raycast.GridRaycaster`. Its source (`volume-cartographer/python/vc/grid_raycast.cpp`) is self-contained, so `windeval/native/CMakeLists.txt` builds it unmodified with nanobind (no OpenMP, which only parallelizes the tree build; results are identical).
2. `vc.Volume.open(...).sample_planes(...)` and `.sample_coords(...)`. `windeval/vcshim/vc/__init__.py` reimplements these in NumPy after reading `python/vc/volume.cpp` and `core/src/render/ChunkedPlaneSampler.cpp` (`sampleCoordsLevelBlockingRequestedLevel`, `samplePinnedTrilinear`): XYZ voxel coordinates, float32 fractions, out-of-array corners read the fill value, a corner in an absent chunk zeroes the sample, clamp then truncate to uint8, validity 1 for every non-sentinel coordinate. Chunks are read from a local zarr directory that fills itself from the bucket on first use (raw 128^3 uint8 chunks, no compressor), so the local copy is a partial mirror of exactly the chunks the slabs touch.

Check: on a real slab, the shim's trilinear output equals `scipy.ndimage.map_coordinates(order=1)` at 99.8 % of 6.3 M voxels and is never more than 1 grey level off (the 1-level cases are float rounding at the truncation step). That confirms axes, origin and interpolation, but not bit equality with the C++ path. A native cross-check needs the macOS bindings from PR #1975.

### Device

Device choice lives in the harness (`windeval/windeval/env.py:pick_device`: CUDA, then MPS, then CPU). The model needed no change for MPS in fp32: the forward pass ran with `PYTORCH_ENABLE_MPS_FALLBACK=0`, so every op is native. DataLoader workers are spawned (see Operational notes for why fork was dropped), so no worker inherits Metal state. villa's own GPU code path for this model is `infer_winding_volume.py`, which pins `cuda:{gpu}`, calls `torch.cuda.set_device` and counts CUDA devices for sharding; patching that is not needed for metrics and could not be tested without a fitted spiral checkpoint, so no villa diff is proposed yet.

### Finding 1: segment coordinates must be in level-2 units

The training config reads PHerc0139 with `volume_scale: 2` and `segment_downscale: 4`. With those values the segment-to-volume transform is the identity, which means Sean's segment coordinates were in 2.399 um voxels divided by 4, and slabs are sampled at spacing 1 in those units (one level-2 voxel, about 9.6 um). The published `-on-20260102150214-2.399um` meshes are in full-resolution voxels. A first smoke test read them with `segment_downscale: 1` (transform 1/4, mathematically consistent) and scored precision 0.13 and count error 2.9, because the slab then samples every full-resolution voxel: the image is zoomed 4x relative to training and every physical length constant (ray length, merge distance, tolerances) shrinks by 4. `scripts/scale_segments.py` writes copies with coordinates divided by 4 (invalid vertices keep -1). With those copies and `segment_downscale: 4`, the smoke test moved into the training range.

### Finding 2: two PHerc0139 segments nearly coincide with their neighbours

Median distance between consecutive winding meshes (2.399 um frame, z 14000 to 16000 full-resolution, nearest-vertex distance) is 50 to 70 voxels for almost every pair. Two pairs break the pattern:

| Pair | Median | 10th percentile | Next pair |
|---|---|---|---|
| w041 to w042 | 15.1 | 7.5 | w042 to w043: 55.7 |
| w045 to w046 | 14.1 | 5.5 | w046 to w047: 157.0 |

In the 9.362 um frame (z 4660 to 5060) the w045 to w046 median is 3.5 voxels with a 10th percentile of 1.0, and both meshes share the same bounding box. (Corrected later the same day: these are nearest-vertex distances. Measured along the sheet normal the pairs are 0.8 and 0.1 level-2 voxels apart, so the dataset's 3-sample crossing merge keeps only one of each pair and the next labelled crossing jumps two indices, which villa masks as a gap. That is why removing w042 and w046 changes nothing below.) Whether Sean's training set contained these segments is unknown.

Other mesh facts: the duplicate `20260130000001-w030_202601301912` segment has no mesh in the 2.399 frame (its original is already in full-resolution coordinates, uuid ends in `_transformed`); the `title_..._flatboi` segment has no winding number and covers z 32939 to 39564 only. Both are left out, leaving 37 segments, w023 to w059.

### Gate configuration

`configs/gate_pherc0139_2399_l2.json`: volume `PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr`, level 2 (19239 x 6628 x 6628, 128^3 uncompressed uint8 chunks); 37 segments; every other dataset and loss setting taken from the checkpoint's own config, except `multiview_fraction` set to 0 (the pair slab is drawn after the primary slab and only feeds the multiview loss, so it does not change the primary slabs or any metric). Ray origins are restricted to level-2 z 3904 to 4288 (about 9 % of the origin quads); all 37 segments stay loaded, so bracket windings and rasterized targets are untouched. Slabs are fresh procedural draws from `WindingModelDataset`, which is exactly what `validation_mode: "separate"` does when no `val_datasets` are given. Losses and metric sums come from `compute_losses` and `accumulate_winding_metrics` in `train_winding_model.py`, imported unchanged; pooled metrics use `finalize_winding_metrics`. Model in eval mode, fp32, batch 2, 8 forked DataLoader workers, seed 1000.

### Provenance note: which checkpoint made the published Paris4 winding products

`https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/winding_inference/manifest.json` (saved as `~/kaggle-data/vesuvius/paris4_winding_inference_manifest.json`) lists `source_attributes.model_ckpt = .../winding_models/runs/winding_model_3d_large_11/ckpt_final.pth`, created 2026-08-14. The released HF checkpoint's config has `out_dir: runs/winding_model_3d_large_11`, so the published Paris4 crossings come from the same run as `winding_model_9um`. This settles the open question in the scoping report (it had only seen the `large_9` example in `NATIVE_PHASE_CACHE.md`). The same manifest names the training CT as `/ephemeral/sean/winding_train_data/PHercParis4/volumes/s1_2um_ds2.zarr`, matching the config.

## Condition 2 preparation (PHerc0139, native 9.362 um 113 keV scan and its Paganin variants)

| Item | URL | Shape (z, y, x), level 0 | Raw size, level 0 |
|---|---|---|---|
| Default reconstruction | `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr` | 20974 x 6621 x 6621 | 919.5 GB |
| Paganin 0 | `.../PHerc0139/volumes/20251107132835-9.362um-1.2m-113keV-pag0-masked.zarr` | 20961 x 6621 x 6621 | 918.9 GB |
| Paganin 50 | `.../PHerc0139/volumes/20251107135911-9.362um-1.2m-113keV-pag50-masked.zarr` | 20961 x 6621 x 6621 | 918.9 GB |
| Meshes in that frame | `.../PHerc0139/segments/<segment>/mesh/<id>-on-20250728140407-9.362um.tifxyz/{x,y,z}.tif, meta.json` | grid step 20 voxels (187 um) | 61.9 MB for 38 segments (37 used) |

All three volumes are OME-Zarr v2, 6 levels, 128^3 chunks of raw uint8 (2 MiB each, no compressor). All three carry the same acquisition (`scanRadix SCROLLS_HEL_4.681um_113keV_1.2m_binmean_2_PHerc_0139_HA`, 2025-07-20). None has a `transform.json`, and there are no `-on-20251107...` meshes. `scripts/check_frame_alignment.py` phase-correlates a 128^3 block of each Paganin variant against the default reconstruction: zero shift at level 2 (correlation 0.92 for pag0, 0.86 for pag50) and zero shift at level 0 (0.80 and 0.76). So the default-frame meshes apply to pag0 and pag50 directly, and their configs omit `volume_id` / `segment_volume_id` (otherwise the dataset would demand a registration file).

Local layout: metadata mirrors in `~/kaggle-data/vesuvius/zarr/PHerc0139/volumes/<name>/` (levels 0 and 2), linked meshes in `~/kaggle-data/vesuvius/eval/PHerc0139/segments_on_20250728140407/` (37 segments, no rescaling needed: `volume_scale: 0`, `segment_downscale: 1`, identity transform). Configs: `configs/cond2_pherc0139_9362_{default,pag0,pag50}.json`, ray origins restricted to z 4660 to 5060 at level 0.

How the z band was chosen: the 2.399 volume's `transform.json` (2.399 base voxels to the 9.362 frame) maps the gate band (full-resolution z 15616 to 17152) to about z 4640 to 5050. (Correction, later the same day: an earlier version of this note said the transform was about 8 voxels off for these meshes. That figure was a nearest-vertex distance into the sparse 9.362 mesh grid, whose vertices are 20 voxels apart, so it measured grid spacing, not misregistration. CT block matching puts `transform.json` within about 1 voxel, and the two published mesh sets agree to 0.74 voxels along the normal; see the condition 2 fair section.)

Two facts that will matter when reading condition 2 against the gate:
- Voxel size is 9.362 um, against about 9.596 um (2.399 x 4) in training, so the model sees sheets about 2.5 % farther apart in samples. Condition 4 (resample to 9.6 um) addresses exactly this.
- The 9.362 meshes are 4 times coarser per axis than the 2.399 meshes (187 um against 48 um grid step), so label positions between grid points are interpolated over longer spans. In the 9.362 frame, w045 and w046 are even closer (median 3.5 voxels) than in the 2.399 frame.

## Gate result, PHerc0139 2.399 um level 2 (run `~/kaggle-data/vesuvius/runs/gate_pherc0139_2399_l2_z3904/`)

Command: `python -m windeval.evaluate configs/gate_pherc0139_2399_l2.json --num-slabs 256 --num-workers 8` (log `~/kaggle-data/vesuvius/logs/gate_model_256.log`). Summary: `python -m windeval.report <run dir>`.

| Metric | This run (256 slabs) | 95 % interval, slab bootstrap | 95 % interval, 8 angular blocks | Training rows: step 49000 | Training rows: last 20 (mean, sd, min to max) |
|---|---|---|---|---|---|
| val_crossing_precision | 0.440 | 0.417 to 0.464 | 0.417 to 0.469 | 0.538 | 0.558, 0.057, 0.439 to 0.644 |
| val_crossing_recall | 0.426 | 0.402 to 0.449 | 0.402 to 0.454 | 0.517 | 0.527, 0.055, 0.413 to 0.612 |
| val_winding_count_error | 0.405 | 0.370 to 0.442 | 0.354 to 0.450 | 0.341 | 0.385, 0.197, 0.219 to 1.191 |
| val_phase_mae | 0.196 | 0.182 to 0.209 | 0.177 to 0.211 | 0.137 | 0.128, 0.045, 0.071 to 0.283 |
| val_density_mae | 0.0144 | | | 0.0094 | |
| val_loss (no multiview term) | 0.052 | | | 0.024 | 0.030 mean |

Counts: 256 slabs (rays), 37 segments loaded, 4,188,680 supervised columns, 39,356,775 labelled crossings. Ray origins drawn from 3,249,496 of 37,708,575 inner quads (z 3904 to 4288 at level 2).

Cost on the M4 Max: 535 s wall for 256 slabs, 2.09 s per slab end to end. The forward pass is the bottleneck at 3.15 s per batch of 2 (1.58 s per slab, p95 3.17 s per batch, fp32 on MPS); 8 forked workers keep the data wait near zero once chunks are cached; villa's metric code takes 0.73 s per batch on CPU. Memory: MPS driver allocation peaked at 12.1 GiB, the main process at 3.0 GiB RSS, the 8 workers at 19.1 GiB RSS combined. Data: 3.07 GB of raw chunks streamed in this run (3.42 GB total in the local mirror, 1,639 chunks of 2 MiB, including the smoke tests).

### Reading of the gate

Count error and its interval overlap the training rows. Precision and recall sit at the bottom edge of the training rows' range and about 0.11 below their mean; phase MAE is about 0.07 higher. Training rows pool three scrolls and only 2 slabs per process per row, so they are a loose reference, but the gap is larger than the run's own interval.

The gap is concentrated in one part of the scroll. Splitting the 256 slabs by whether their labelled winding bracket includes w042 or w046:

| Slabs | n | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|---|
| bracket includes w042 or w046 | 129 | 0.355 | 0.342 | 0.509 | 0.258 |
| bracket includes neither | 127 | 0.556 | 0.541 | 0.298 | 0.132 |

The clean half matches the training rows (0.558, 0.527, 0.385, 0.128). Longer brackets are more likely to include w042 or w046, so the split was repeated within bracket-length bins; the gap holds in every bin:

| Windings in bracket | Clean: n, P, R, count error | Includes w042 or w046: n, P, R, count error |
|---|---|---|
| 4 to 6 | 57, 0.623, 0.603, 0.259 | 22, 0.389, 0.349, 0.465 |
| 7 to 12 | 63, 0.521, 0.506, 0.325 | 45, 0.305, 0.294, 0.572 |
| 13 or more | 7, 0.547, 0.543, 0.376 | 62, 0.373, 0.362, 0.479 |

Verdict, first pass: the harness reproduces the training-batch numbers on the training distribution in the part of the scroll outside windings 41 to 47, so the port (shim, rescaled meshes, MPS) is not what lowers the pooled number. The pooled PHerc0139 band is lower because of that part of the scroll. See the follow-up runs below for what is and is not behind it.

## Follow-up runs on the gate band

### Without w042 and w046 (`runs/gate_pherc0139_2399_l2_z3904_nodup/`)

Same config with 35 segments. Pooled: precision 0.439 (0.420 to 0.458), recall 0.419 (0.400 to 0.439), count error 0.405 (0.373 to 0.438), phase MAE 0.188 (0.176 to 0.200). Dropping the two near-duplicates changes almost nothing. Slabs whose bracket covers windings 41 to 47 are still worse (P 0.376, R 0.356, CE 0.465, n 144) than slabs that avoid them (P 0.544, R 0.527, CE 0.328, n 112). So my first explanation was wrong: the near-duplicates are a real data issue but not the cause of the gap. The weak region is windings 40 to 47 themselves, with or without w042 and w046.

A second hypothesis, to be tested with the per-winding diagnostic added to `evaluate.py` (fraction of each winding's labelled crossings that have a decoded passage within the 2-sample tolerance, and the mean signed offset): most of w040 to w045 were traced on the 9.362 um scan (their `tifxyz_original` meta says `volume: 4.681um_113keV_1.2m_binmean_2_PHerc_0139_110_surf`), and their `-on-20260102150214` meshes were carried into the 2.399 frame. Issue #1912 found meshes carried between PHerc0139 scans sitting up to about 100 um off the inked surface for w043, w044 and w045. A 2-slab smoke test of the diagnostic already shows w041, w043, w044 and w045 with almost no labels within tolerance and passages 3 to 4 samples toward lower windings, but 2 slabs prove nothing; the 256-slab rerun is queued.

### Numerical checks

- CPU against MPS, fp32, same 2 slabs (seed 1000, no workers): identical metrics to 4 decimals (P 0.3283, R 0.3151, CE 0.3793, phase MAE 0.2616). CPU forward took 34.6 s per batch of 2 against 3.2 s on MPS (about 11x), measured while another MPS run was active, so treat the CPU figure as rough.
- `tests/test_vcshim.py` (5 tests, synthetic arrays): trilinear agrees with scipy within one grey level, integer coordinates read exact voxels, out-of-array points read the fill value and stay valid, sentinel coordinates are invalid, a corner in an absent chunk zeroes the sample, and `sample_planes` lays planes out as [plane, y step, x step].

### Operational notes

- The first PHerc1667 run died at start: macOS aborted the forked DataLoader workers with `+[Swift.__SharedStringStorage initialize] may have been in progress in another thread when fork() was called`. With `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` the retry died with a worker segfault instead. The PHerc0139 runs had not hit it, so it is a race at fork time, and disabling the check is not a fix. `evaluate.py` now defaults to spawned workers: `windeval/lazy.py` wraps the dataset so each worker builds its own `WindingModelDataset` from the config (the nanobind raycasters cannot be pickled). Fork and spawn draw identical slabs with identical metric sums (checked on 8 slabs, 2 workers), because sampling uses only the per-worker torch RNG that the DataLoader seeds. `--start-method fork` remains available.
- `vesuvius.tifxyz` caches decoded coordinate TIFFs under `tempfile.gettempdir()`. The first smoke test wrote 533 MB there; it was deleted, and `env.setup()` now points `TMPDIR` and `tempfile.tempdir` at `~/kaggle-data/vesuvius/tmp`. The rescaled meshes are written uncompressed, so the reader maps them directly and makes no cache copies. pytest scratch goes to the same place (`tests/conftest.py`).

### EMA weights, bf16, and determinism (same 256 slabs as the gate)

Re-running the gate config with the same seed and worker count draws the same 256 slabs (origins and label counts identical, 256 of 256), so the following are paired comparisons.

| Run | Precision | Recall | Count error | Phase MAE | Forward per batch of 2 | MPS driver peak |
|---|---|---|---|---|---|---|
| `model`, fp32 (gate) | 0.440 | 0.426 | 0.405 | 0.196 | 3.15 s | 12.1 GiB |
| `model_ema`, fp32 | 0.441 | 0.426 | 0.405 | 0.196 | 3.15 s | 12.1 GiB |
| `model`, bf16 autocast (as in training) | 0.446 | 0.429 | 0.394 | 0.194 | 1.48 s | 10.2 GiB |

EMA and online weights are interchangeable here (the learning rate had decayed to about 5e-7 by the end). bf16 autocast on MPS is 2.1 times faster than fp32 with no metric loss; fp32 stays the reference setting.

### Where the model and the labels disagree, winding by winding

From the bf16 run (the diagnostic was added before it started): for each winding, the share of its labelled crossings that have a decoded passage within the 2-sample tolerance, and the mean signed offset (positive means the passage sits toward higher windings).

| Windings | Share within tolerance | Mean signed offset |
|---|---|---|
| w024 to w033 | 0.46 to 0.66 | -0.1 to +1.2 |
| w034 to w039 | 0.36 to 0.46 | -0.9 to +0.3 |
| w040 to w049 | 0.21 to 0.39 (w047 0.22, w048 0.21) | -1.6 to +1.3 |
| w050 to w059 | 0.40 to 0.58 | -1.5 to +0.8 |

So the trouble is a band of windings, w040 to w049, with no systematic shift (offsets stay within 1.6 samples). The "traced on the 9.362 scan" idea does not hold: w056 to w058 were traced there and agree well (0.54 to 0.58), and w046 to w049 were traced on the 2 um data and agree worst. villa's own slab plots for rays started on w047 and w048 (`~/kaggle-data/vesuvius/runs/diagnose/gate_pherc0139_2399_l2_z3904_src47_48/slab00.png`, `slab01.png`) show labelled crossings arriving in close pairs (4 to 9 samples apart) separated by wide gaps, while the model spaces its passages evenly at 12 to 13 samples. The CT in those stretches is dense and disordered. Whether the pairs are two faces of one delaminated sheet labelled as two windings, or real tightly packed windings, cannot be settled from these plots; it needs someone who knows these segments. Either way, this band, not the port, accounts for the pooled PHerc0139 numbers sitting below the training rows.

## Condition 2, first pass (PHerc0139 native 9.362 um, 113 keV, three reconstructions)

Same model, fp32, 256 slabs each, seed 1000, ray origins z 4660 to 5060 at level 0, 37 segments meshed in the 9.362 frame. The three runs draw nearly the same slabs (225 of 256 share their geometry between default and pag0; draws diverge only where the CT-is-nonzero check on a crossing differs between reconstructions), so they are close to paired. Gate numbers repeated for reference; the gate uses the finer 2.399-frame meshes and a different label set, so the gate-to-default difference mixes scan and mesh effects.

| Volume | Phase retrieval (from metadata.json) | Precision | Recall | Count error | Phase MAE | val_loss |
|---|---|---|---|---|---|---|
| 2.399 um, level 2 (gate) | Paganin delta/beta 1000, unsharp 4 / 1.2 | 0.440 (0.417 to 0.464) | 0.426 (0.402 to 0.449) | 0.405 (0.370 to 0.442) | 0.196 | 0.052 |
| 9.362 um default `20250728140407` | Paganin delta/beta 1000, unsharp 4 / 1.2 | 0.422 (0.398 to 0.447) | 0.389 (0.364 to 0.414) | 0.520 (0.461 to 0.584) | 0.243 | 0.067 |
| 9.362 um `pag50` | Paganin delta/beta 50, unsharp 4 / 1.2 | 0.264 (0.252 to 0.276) | 0.274 (0.262 to 0.284) | 2.191 (1.843 to 2.537) | 0.728 | 0.192 |
| 9.362 um `pag0` | delta/beta 0.0001 (none), no unsharp | 0.266 (0.253 to 0.279) | 0.263 (0.251 to 0.275) | 2.049 (1.730 to 2.369) | 0.677 | 0.180 |

Intervals are 95 % slab bootstraps. Each run: about 4.18 M supervised columns, 37.8 to 39.8 M labelled crossings, 3.58 to 3.60 GB of chunks streamed per volume (about 1,730 chunks), 640 to 700 s wall on MPS.

First reading, to be firmed up:
- On the eligible-protocol scan with the same phase retrieval as training (delta/beta 1000 plus unsharp masking), the model holds up: precision 0.42 and recall 0.39 against 0.44 and 0.43 on the training scan, with count error rising from 0.41 to 0.52. Part of that drop may come from the coarser 9.362 meshes (187 um grid against 48 um), not from the scan. Per winding, the inner windings w023 to w039 agree as well or better than on the 2.399 scan, the w040 to w049 band is weak on both, and the outer windings w050 to w059 drop (agreement 0.23 to 0.51 against 0.40 to 0.58).
- The two Paganin variants break the model: count error rises about four-fold to above 2 windings per column, and phase MAE triples. Both use a far weaker phase retrieval than training (delta/beta 50 and essentially none). villa's slab plot for a pag0 slab (`~/kaggle-data/vesuvius/runs/diagnose/cond2_pherc0139_9362_pag0_z4660/slab01.png`) shows the edge-enhanced texture of weak phase retrieval. So `winding_model_9um` is tied to the delta/beta 1000 reconstruction, and any scan for spiral fitting should be reconstructed that way or the model retrained with these variants. The failures concentrate in a few blocks (angular sector 4 has count error near 5), so the block interval for count error is wide (1.2 to 3.5).
- Voxel size (9.362 against 9.596 um) is not separated from the mesh change here; condition 4 (resampling to 9.6 um) is the planned test.

### Reproducibility

The gate run was repeated after the switch to spawned workers (`runs/gate_pherc0139_2399_l2_z3904_rerun/`): all 256 per-slab metric sums are bit-identical to the first (forked) run, so MPS fp32 inference and the sampling are deterministic for a given seed and worker count. The rerun also carries the per-winding table in `result.json` (fp32 values within 0.02 of the bf16 table above).

## Second training scroll: PHerc1667 2.399 um level 2 (`runs/gate_pherc1667_2399_l2_z5000/`)

Volume `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr` level 2 (9269 x 3808 x 3808), meshes `mesh/<id>-on-20251217075048-2.399um.tifxyz` divided by 4 (`eval/PHerc1667/segments_on_20251217075048_ds4_single/`, 19 segments). Left out: `w011_..._merged_v4_flatboi_straightened_v4` (a 2061 x 30097 grid that spans many windings under the single name w011, which the dataset would label as one winding). Ray origins z 5000 to 5384.

| Metric | PHerc1667 | PHerc0139 (gate) | Training rows, last 20 mean |
|---|---|---|---|
| Precision | 0.446 (0.420 to 0.472) | 0.440 | 0.558 |
| Recall | 0.414 (0.393 to 0.435) | 0.426 | 0.527 |
| Count error | 0.506 (0.416 to 0.600) | 0.405 | 0.385 |
| Phase MAE | 0.187 (0.164 to 0.211) | 0.196 | 0.128 |

Per winding, w028 to w034 agree at 0.40 to 0.48 within tolerance, while w011 to w013 and w023 (0.01 to 0.13) and w040 to w041 (0.02 to 0.14) barely agree. PHerc1667's labelled windings have gaps (w013, w018, w023, then w028 to w041), so many brackets carry index jumps of 5 that the dataset masks out. 2.26 GB streamed, 645 s wall.

## Gate decision

**Pass, with one stated caveat.** Evidence that the harness is a faithful port: the shim matches an independent trilinear sampler, MPS fp32 matches CPU fp32 to 4 decimals, the run is bit-reproducible, EMA and online weights agree, bf16 changes nothing material, and on PHerc0139 slabs that avoid windings 40 to 49 the four metrics (0.556, 0.541, 0.298, 0.132) sit on the training rows' means (0.558, 0.527, 0.385, 0.128). The caveat: pooled over a whole z band, both training scrolls score about 0.44 precision and 0.42 recall, at the bottom of the training rows' range. The pooled gap traces to specific winding bands where the model and the published labels disagree, and the training rows also pool PHercParis4 spiral-fit data, which is not public (the config reads `/ephemeral/sean/winding_train_data/PHercParis4/spiral_fit`) and so could not be tested. The real held-out comparison (condition 3, PHerc0841) should be read against these in-distribution numbers from this harness, not against `metrics.jsonl`.

## Totals for day 1

| Item | Amount |
|---|---|
| CT chunks streamed (raw 128^3 uint8) | 17.3 GB: PHerc0139 2.399 L2 4.17 GB (1,989 chunks), 9.362 default 3.61 GB, pag0 3.63 GB, pag50 3.62 GB, PHerc1667 2.399 L2 2.26 GB (1,076 chunks) |
| Meshes downloaded | 2.73 GB (PHerc0139 1.10 GB, PHerc1667 1.62 GB) |
| Checkpoint and metadata | 0.15 GB |
| Derived meshes written (level-2 copies) | 2.4 GB in `~/kaggle-data/vesuvius/eval/` |
| Evaluation runs | 10 full runs of 256 slabs (2,560 slabs), each 6 to 12 min on MPS |

## Next steps

1. Condition 3 (unseen scroll, PHerc0841): list its segments and mesh frames, check that they carry winding numbers, run its 9.366 um and 2.403 um scans with the same harness.
2. Condition 2, cleaner: carry the fine 2.399-frame meshes into the 9.362 frame with an affine fitted between the two published mesh sets (the published `transform.json` is about 8 voxels off for these meshes), so the scan effect is measured with the same labels; then condition 4 (resample 9.362 to 9.6 um).
3. A second z band per scroll and 512 slabs per condition for tighter intervals.
4. Ask the team (or check VC3D) whether the PHerc0139 w041/w042 and w045/w046 pairs are duplicates, and what the paired labels in the w040 to w049 band are.
5. If a villa change is wanted: an evaluation-only entry point for `train_winding_model.py` (run `validation_mode: "separate"` from a checkpoint without training), with the device choice used here. `infer_winding_volume.py` would need more than a device swap on MPS (`torch.cuda.set_device`, CUDA device counting for shards), and it cannot be tested without a fitted spiral checkpoint.

## How to run

```
cd windeval   # this directory
../.venv/bin/python -m pytest -q tests/
../.venv/bin/python -m windeval.evaluate configs/gate_pherc0139_2399_l2.json --num-slabs 256 --num-workers 8
../.venv/bin/python -m windeval.report ~/kaggle-data/vesuvius/runs/gate_pherc0139_2399_l2_z3904
../.venv/bin/python -m windeval.diagnose configs/gate_pherc0139_2399_l2.json --num-slabs 3
```

Setup steps, in order (see also README.md): `scripts/list_segments.py` and `scripts/fetch_meshes.py` (meshes), `scripts/make_mirror.py` (volume metadata; chunks then stream on demand), `scripts/link_segments.py` and `scripts/scale_segments.py` (segment folders the dataset can read), and the grid_raycast build: `cmake -S native -B ~/kaggle-data/vesuvius/build/grid_raycast -G Ninja -DPython_EXECUTABLE=../.venv/bin/python && cmake --build ~/kaggle-data/vesuvius/build/grid_raycast`, then copy the `.so` into `vcshim/vc/`. Long runs go through `scripts/run_queue.sh QUEUE_FILE [PID_TO_WAIT_FOR]` with logs in `~/kaggle-data/vesuvius/logs/`.

## 2026-10-08, after the gate: conditions 2 (fair), 3 and 4

### Fitted transform, 2.399 um level 2 to 9.362 um (`results/transform_pherc0139_2399l2_to_9362.json`)

`windeval/fit_transform.py` (spec `fits/pherc0139_2399l2_to_9362.json`): 300 blocks of 48^3 voxels centred on 2.399-frame mesh vertices in the gate band, each carried into the 9.362 default scan by the published `transform.json`, then phase-correlated against the 2.399 level-2 scan resampled onto the same grid. 299 blocks matched with NCC above 0.3 (median NCC 0.94), 293 inliers after robust refitting. Residuals in 9.362 voxels: published transform 1.05 median (1.53 p90); fitted affine 0.22 median (0.38 p90); two-fold held-out 0.22 median (0.40 p90), i.e. about 2 um. One level-2 unit of the 2.399 scan is 1.0287 voxels of the 9.362 scan (9.63 um, against 9.596 um nominal).

With this affine, the published 9.362 meshes sit 0.74 voxels (median over segments, p90 1.85) from the registered 2.399 meshes along the sheet normal; the 8-voxel figure in my earlier note was grid spacing, not offset.

### Condition 2, fair version

`windeval/replay.py` draws every slab from the 2.399 level-2 dataset (rays, brackets, observability and targets unchanged) and re-samples only the image from the 9.362 scan through the fitted affine, at the same physical sample spacing. All 256 rays and label counts are identical to the gate run (checked slab by slab), and on three test slabs the replayed image correlates with the reference image at 0.95 to 0.97, peaking at zero shift on all three axes. Results (256 slabs each, 95 % slab bootstrap, change against the 2.399 run with a paired bootstrap over the same slabs):

| Scan (same rays and labels) | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| 2.399 um level 2 (reference) | 0.440 | 0.426 | 0.405 | 0.196 |
| 9.362 um default (delta/beta 1000) | 0.409, change -0.032 (-0.047 to -0.017) | 0.381, -0.044 (-0.061 to -0.029) | 0.518, +0.113 (+0.060 to +0.164) | 0.240, +0.044 (+0.031 to +0.057) |
| 9.362 um pag50 | 0.269, -0.171 | 0.267, -0.158 | 1.904, +1.499 (+1.190 to +1.833) | 0.659, +0.463 |
| 9.362 um pag0 | 0.273, -0.167 | 0.259, -0.167 | 1.819, +1.415 (+1.125 to +1.705) | 0.625, +0.430 |

The same pattern holds on both sides of the w040 to w049 split (REPORT.md has the split tables). The first-pass condition 2 numbers (coarse 9.362 meshes, 0.422 / 0.389 / 0.520 / 0.243) are within noise of the fair default run, so the coarser meshes were not what cost accuracy there; the scan was.

A second fit iterated four times (`results/transform_pherc0139_2399l2_to_9362_iterated.json`) because windowed phase correlation under-reads sub-voxel shifts: under the iterated fit the measured block shifts are 0.10 voxels median (p90 0.23, held-out 0.10). It differs from the first-pass fit used by all replay runs by 0.09 voxels median and 0.27 at most over the band, about 2.5 um, far below the 2-sample (19 um) matching tolerance, so the runs were not repeated.

### Condition 3: PHerc0841 is blocked by its labels

PHerc0841 has three published segments: `w00` and two `auto_grown_*` surfaces. Each covers 0.77 to 0.78 of a turn (`_vertex_turns` range), they are distinct sheets (22 to 52 voxels apart in the 9.366 frame), and only `w00` has a winding number. villa's sampler needs either two labelled windings at least `min_winding_gap` = 4 apart (`_labelled_ray`) or one segment crossed twice with different unwrapped turns (`_multi_wrap_ray`). `windeval/count_rays.py` over the whole scroll, no region limit: 0 rays from 3,000 builder attempts and 0 samples from 8 draws of 64 attempts each, on both the 9.366 um and the 2.403 um scans. Widening the region cannot help; the segment set has no multi-winding supervision.

Substitutes found by listing the bucket for winding-named segments outside the three training scrolls:
- `PHercMANBp`: 11 segments, w0 to w8 (w6 and w7 twice), only `intermediate/tifxyz_original` meshes, in full-resolution voxels of the 2.399 um 78 keV scan `20251216152116` (same protocol as training). A CT profile along the mesh normal shows the same edge pattern as PHerc0139's meshes, so they are in that scan's frame. 332 of 600 builder attempts give rays (4 to 9 crossings).
- `PHerc0172` (Scroll 5): 44 consecutive windings w052 to w095 (plus 9 unlabelled `auto_grown` pieces) on a 7.91 um 53 keV scan (`20241024131839`, uncompressed export; the `...838` export is blosc-compressed, which the NumPy sampler does not read). Different beamline, energy and voxel size. Run twice: at native 7.91 um sampling, and resampled to 9.596 um by dividing the meshes by 1.213148 and mapping back with `segment_to_volume_scale` (a fractional transform villa's integer `segment_downscale` cannot express). 252 and 296 of 600 builder attempts give rays (median 10 to 11 crossings).

### Label pairs, sharper

Measured along the sheet normal instead of nearest-vertex distance (`scripts/winding_spacing.py`, gate band, level-2 voxels): consecutive windings sit 10 to 17 voxels apart, but w041 to w042 is 0.8 (p10 0.1) and w045 to w046 is 0.1 (p10 0.0). In this band w042's mesh lies on w041's sheet and w046's on w045's. w046 to w047 is 30.5, about two gaps, and w039 to w040 is 24.1. A renumbered label set (drop w042 and w046, shift w043 and above down by one) tests whether a numbering offset explains the weak w040 to w049 band: it does not. Pooled precision 0.399, recall 0.389, count error 0.635; slabs touching the band fall to precision 0.319 and count error 0.861 (from 0.375 and 0.488 with the published numbers), while slabs avoiding it stay at 0.574 / 0.557 / 0.276. The published numbering is the more consistent one; w042's and w046's meshes are misplaced onto their neighbours' sheets in this band, and the sheets that should carry w042 and w046 have no mesh here. The band's low agreement is left unexplained by labels alone.

### villa patches (`patches/`)

Three diffs against `e0bbb8b`, each applying cleanly on its own, tested in a local worktree (`~/kaggle-data/vesuvius/villa-windeval`, branch `windeval-macos`, never pushed): picklable dataset plus `dataloader_start_method` (spawn workers), `segment_coordinate_divisor` (read published full-resolution meshes directly), and a zarr-based NumPy fallback for `vc.Volume`. `scripts/check_villa_patches.py` draws 8 PHerc0139 slabs through the patched villa (villa's own dataset pickled into spawn workers, published meshes with divisor 4, only the standalone `vc.grid_raycast` present) and through the harness: labels identical and all 50.3 M image voxels identical.

### Condition 4 (PHerc0139, same rays as the fair condition 2 runs)

| Variant of the 9.362 default scan | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| sampled at 9.63 um (one 2.399 level-2 unit; fair condition 2) | 0.409 | 0.381 | 0.518 | 0.240 |
| native 9.36 um sampling (labels re-rendered for that frame, same rays) | 0.409, change -0.000 (-0.004 to +0.004) | 0.380, -0.001 | 0.517, -0.001 (-0.014 to +0.012) | 0.244, +0.004 |
| Gaussian blur along the ray, sigma 1 sample | 0.348, -0.061 | 0.284, -0.098 | 1.222, +0.705 (+0.581 to +0.843) | 0.427, +0.187 |

The 2.8 % voxel-size gap does not matter; blurring across the sheets hurts. A blur along the scan's z axis (along the sheets) is queued.

### Condition 3 results

| Scroll and scan | Precision | Recall | Count error | Phase MAE | Notes |
|---|---|---|---|---|---|
| PHerc0841, 9.366 um and 2.403 um | no rays | | | | 0 of 3,000 ray attempts per scan; segments carry no multi-winding labels |
| PHercMANBp, 2.399 um 78 keV level 2 | 0.349 (0.334 to 0.365) | 0.274 (0.259 to 0.289) | 0.610 (0.546 to 0.673) | 0.196 (0.183 to 0.210) | 11 segments, w0 to w8; per-winding agreement 0.18 to 0.35 |
| PHerc0172, 7.91 um 53 keV, native sampling | 0.251 (0.243 to 0.259) | 0.140 (0.129 to 0.151) | 4.854 (4.408 to 5.332) | 1.502 (1.381 to 1.620) | 44 segments, w052 to w095 |
| PHerc0172, resampled to 9.596 um | 0.306 (0.298 to 0.315) | 0.324 (0.314 to 0.335) | 2.447 (2.188 to 2.716) | 0.824 (0.757 to 0.891) | per-winding agreement 0.24 to 0.43, flat across windings |

PHerc0172 is a Diamond Light Source scan with no reconstruction metadata in the bucket, so its phase-retrieval setting is unknown. Resampling to the training voxel size halves its count error and more than doubles recall, but it stays far from the ESRF scans. MANBp, on the training protocol, sits between the training scrolls and PHerc0172; with only nine windings near the scroll's core its labels and geometry differ from the training bands, so this is one data point, not a protocol verdict. The two PHerc0172 runs are not on identical rays (segment units change the sampler's constants), so their comparison is unpaired.

## Deliverables status (2026-10-08, 06:20 CDT)

- Fair condition 2, condition 3 (with substitutes for PHerc0841), condition 4 (resampling, z-axis blur, ray-axis blur) and the label tests are complete; tables are in `results/compare_*.md`, figures in `figures/`.
- `MODEL_CARD.md`, `README.md`, `patches/` (three diffs plus rationale), `LICENSE` (MIT), `requirements.txt`, `.gitignore` are in place. No absolute user paths remain in code, configs or results (`scripts/collect_results.py` rewrites them to `${WINDEVAL_DATA}`).
- The compiled `vcshim/vc/grid_raycast*.so` is a local build artifact (ignored by `.gitignore`); `scripts/build_grid_raycast.sh` recreates it.
