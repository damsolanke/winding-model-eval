---
library_name: pytorch
tags:
- vesuvius-challenge
- herculaneum-papyri
- x-ray-ct
- winding-number
---

# winding_model_9um

Draft model card, written from the checkpoint, its config and an independent evaluation ([windeval](README.md)). It is offered to the maintainers as a starting point; everything marked "not stated" needs their input.

## Summary

A 9.1 M parameter network that reads a 128 x 128 x 384 voxel CT slab around a ray through a rolled Herculaneum scroll and predicts, for every column of voxels parallel to the ray, a monotone relative winding phase. Integer passages of the phase mark where the column crosses a papyrus winding; phase increments are winding-density integrals. These are the observations `fit_spiral` consumes. The published PHercParis4 winding inference (`spiral_datasets/PHercParis4/winding_inference/`, 2026-08-14) names this run's final checkpoint (`winding_model_3d_large_11/ckpt_final.pth`) as its source.

## Model details

| Item | Value |
|---|---|
| Code | `villa/vesuvius/src/vesuvius/neural_tracing/winding_models/winding_model.py` (`WindingModel`) |
| Checkpoint | `ckpt_final.pth`, 146,441,803 bytes, sha256 `805a868c5ef518e4edb567ac2ddc63dfbf870f21b1164ae1642baa1781ad8f02`; keys `model`, `model_ema`, `optimizer`, `lr_scheduler`, `config`, `step` (50000) |
| Parameters | 9,135,724 (fp32) in both `model` and `model_ema`; villa's inference script loads `model` |
| Architecture | 3D conv encoder (32, 64, 96 channels, each stage halves the two transverse axes only), 2 residual blocks at width 192, 6 axial attention blocks (6 heads; attention along the ray with relative bias up to 128, and over the 16 x 16 transverse plane), one decoder level (96), full-resolution head; no crossing head, no variance head |
| Input | slab `[B, 128, 128, 384]` of uint8 CT plus a validity mask; per-slab standardisation inside the model. One sample is one voxel of the training scans' level 2, about 9.6 um |
| Output | `phase` and `phase_increments`, `[B, 128, 128, 384]` (column stride 1) |
| Decoding | crossings are the integer passages of the phase after one free offset per slab |
| Compute | fp32 forward of a batch of 2: 3.2 s on an Apple M4 Max (MPS), 12 GiB peak; bf16 autocast 1.5 s with the same metrics |

## Intended use

Dense per-ray winding observations for global spiral fitting of whole scrolls (`infer_winding_volume.py` then `fit_spiral`). Not a segmentation or ink model.

## Training data (from the checkpoint's config)

| Scroll | Volume | Read at | Segments |
|---|---|---|---|
| PHerc0139 | `20260102150214-2.399um-0.2m-78keV-masked.zarr` | level 2 (`volume_scale: 2`), meshes at full resolution / 4 (`segment_downscale: 4`) | `winding_train_data/PHerc0139/segments` (not published as such; the public `-on-20260102150214-2.399um` meshes divided by 4 reproduce the training numbers, see below) |
| PHerc1667 | `20251217075048-2.399um-0.2m-78keV-masked.zarr` | level 2, meshes / 4 | `winding_train_data/PHerc1667/segments` |
| PHercParis4 | `s1_2um_ds2.zarr` | scale 0 | `spiral_fit` (not public) |

All three training CT volumes were reconstructed with Paganin phase retrieval at delta/beta 1000 plus unsharp masking (coefficient 4, sigma 1.2), according to the public volumes' `metadata.json`. Rays are sampled from labelled segments (numbered windings at least 4 apart, or multi-wrap segments), proportional to segment area. The fraction of slabs drawn from each scroll is not stated.

## Training procedure (from config)

50,000 iterations, warm start (weights only) from `winding_model_3d_large_8/ckpt_020000.pth`; batch 2 per process (process count not stated); AdamW, learning rate 5e-4, weight decay 0.01, cosine schedule with 500 warm-up steps; bf16 mixed precision; gradient clip 1.0; EMA 0.999; `torch.compile`. Losses: phase (Huber 0.25, weight 1), density (weight 1), span delta (0.2), multiview consistency (0.25, half of the samples carry a second slab within a 30 degree cone), position hinge and snap (0.2 each). Seed 39.

## Evaluation

### Metrics

All from `train_winding_model.py`: **crossing precision / recall** (decoded passages matched one to one to labelled crossings within 2 samples), **winding count error** (absolute difference between predicted and labelled phase change over each supervised column), **phase MAE** (after one free offset per slab). 

### What the published metrics are

`metrics.jsonl` (50 rows, steps 500 to 49,000) was logged with `validation_mode: "train_batch"`: every `val_*` value is computed on the training batch of that step, pooled over the three training scrolls. Last row: precision 0.538, recall 0.517, count error 0.341, phase MAE 0.137; last 20 rows mean 0.558, 0.527, 0.385, 0.128.

### Fresh draws on the training scrolls

256 slabs each, `separate`-style fresh draws, fp32, 95 % slab-bootstrap intervals.

| Data | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| PHerc0139, 2.399 um level 2, all slabs | 0.440 (0.417 to 0.464) | 0.426 (0.402 to 0.449) | 0.405 (0.370 to 0.442) | 0.196 (0.182 to 0.209) |
| same, slabs avoiding windings w040 to w049 (94) | 0.588 (0.556 to 0.620) | 0.573 (0.541 to 0.604) | 0.260 (0.223 to 0.299) | 0.116 (0.102 to 0.131) |
| same, slabs touching w040 to w049 (162) | 0.375 (0.347 to 0.401) | 0.361 (0.333 to 0.388) | 0.488 (0.444 to 0.535) | 0.242 (0.225 to 0.258) |
| PHerc1667, 2.399 um level 2 | 0.446 (0.420 to 0.472) | 0.414 (0.393 to 0.435) | 0.506 (0.416 to 0.600) | 0.187 (0.164 to 0.211) |

EMA weights and bf16 autocast give the same numbers within 0.01.

### Same rays, other reconstructions (PHerc0139)

Identical rays and labels; only the CT changes (fine 2.399 um meshes carried into the 9.362 um frame by a CT-fitted affine, 0.10 voxel median residual).

| CT | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| 2.399 um, 78 keV, 0.22 m (training scan) | 0.440 | 0.426 | 0.405 | 0.196 |
| 9.362 um, 113 keV, 1.2 m, delta/beta 1000 | 0.409 | 0.381 | 0.518 | 0.240 |
| same scan, Paganin delta/beta 50 (`pag50`) | 0.269 | 0.267 | 1.904 | 0.659 |
| same scan, no phase retrieval (`pag0`) | 0.273 | 0.259 | 1.819 | 0.625 |

### Unseen scrolls

| Scroll and scan | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| PHercMANBp, 2.399 um 78 keV level 2 (training protocol), windings 0 to 8 | 0.349 (0.334 to 0.364) | 0.274 (0.260 to 0.288) | 0.610 (0.548 to 0.673) | 0.196 (0.183 to 0.210) |
| PHerc0172 (Scroll 5), 7.91 um 53 keV DLS, native sampling | 0.251 (0.243 to 0.260) | 0.140 (0.129 to 0.151) | 4.854 (4.394 to 5.296) | 1.502 (1.380 to 1.622) |
| PHerc0172, resampled to 9.596 um | 0.306 (0.298 to 0.314) | 0.324 (0.314 to 0.335) | 2.447 (2.178 to 2.714) | 0.824 (0.758 to 0.895) |

PHerc0841 could not be scored: its three published segments are single sheets of less than one turn, and only one carries a winding number, so villa's sampler finds no supervised rays (0 of 3,000 attempts per scan).

## Known failure modes and limitations

- **Phase retrieval.** On reconstructions with weaker phase retrieval than training (delta/beta 50 or none) the winding count error rises from 0.5 to about 1.9 windings per column. Use delta/beta 1000 reconstructions, or retrain with these variants.
- **Scan protocol.** On the eligible-protocol 9.362 um 113 keV scan of the same sheets the model loses about 0.03 precision and 0.04 recall, and count error rises by 0.11. Resampling that scan to the training voxel size changes nothing measurable.
- **Voxel size.** At native 7.91 um sampling (PHerc0172) count error is 4.9 windings per column; resampling to 9.6 um halves it. A 2.8 % mismatch (9.36 against 9.63 um) has no measurable effect. Gaussian blur along or across the sheets makes things worse.
- **Other beamlines.** On the DLS scan of PHerc0172 the model stays well below its ESRF numbers even after resampling (count error 2.4).
- **Labels.** In PHerc0139 the meshes named w042 and w046 lie on the sheets of w041 and w045 (0.8 and 0.1 level-2 voxels apart along the normal in the evaluated band), and w046 to w047 spans about two winding gaps. Removing them changes nothing measurable (the sampler merges coincident crossings), but anything that reads them as distinct windings gets two numbers for one sheet. Slabs crossing windings w040 to w049 score far below the rest for reasons these two meshes do not explain.
- **Single checkpoint, single band per scroll.** Numbers come from one z band per scroll (256 slabs each); blocks of the scroll vary (angular-sector intervals are reported in windeval).
- **Inputs must match the training sample size.** Segment coordinates must be in units of the training voxel (about 9.6 um). The published full-resolution meshes need dividing by 4 for level-2 slabs; read as-is, precision falls to 0.13.

## How to reproduce

See [REPORT.md](REPORT.md) (exact commands) and [README.md](README.md) (setup). The evaluation reuses villa's dataset, losses and metric code unchanged.

## License

Not stated in the Hugging Face repository.
