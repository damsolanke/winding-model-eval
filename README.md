# windeval

Held-out evaluation of the Vesuvius Challenge winding model `scrollprize/winding_model_9um` on real scroll data, run on an Apple M4 Max (MPS). It scores the released checkpoint with villa's own dataset, losses and metrics (`train_winding_model.py`, `validation_mode: "separate"` semantics) on scans and scrolls the training log never reported.

Results are in [REPORT.md](REPORT.md); a draft model card is in [MODEL_CARD.md](MODEL_CARD.md); working notes are in [NOTES.md](NOTES.md).

## What is in here

| Path | Purpose |
|---|---|
| `windeval/evaluate.py` | Draw slabs with villa's `WindingModelDataset`, run the model, pool villa's metric sums, keep per-slab sums for bootstrap intervals |
| `windeval/replay.py` | Score a second scan on exactly the rays and labels of a reference scan (fitted affine between the two voxel frames) |
| `windeval/fit_transform.py` | Fit that affine by CT block matching (phase correlation), with held-out residuals |
| `windeval/compare.py`, `windeval/report.py` | Paired tables with bootstrap intervals, split by winding band; run summaries |
| `windeval/count_rays.py` | How many supervised rays a segment set yields under villa's sampling rules |
| `windeval/diagnose.py` | villa's slab plots and label-to-passage residuals for a few slabs |
| `vcshim/vc/` | NumPy stand-in for the two `vc.Volume` calls the dataset makes (villa's bindings do not build on macOS) |
| `native/` | CMake build of villa's unmodified `vc.grid_raycast` extension |
| `patches/` | Minimal villa diffs for macOS (spawn workers, mesh-unit divisor, NumPy sampling fallback), each with its rationale |
| `configs/`, `fits/` | Run and fit configs; data paths use `${WINDEVAL_DATA}` |
| `results/` | Fitted transforms, comparison tables and per-run results (numbers only, no CT data) |
| `figures/` | Figures used in REPORT.md (no CT pixels) |
| `scripts/` | Data preparation, queueing, checks |

## Setup

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
git clone https://github.com/ScrollPrize/villa "$WINDEVAL_DATA/villa"
git -C "$WINDEVAL_DATA/villa" checkout e0bbb8b40a2db58b1d71864f286eb85717e59e64
PYTHON=.venv/bin/python scripts/build_grid_raycast.sh
.venv/bin/python -m pytest -q tests/
```

`WINDEVAL_DATA` (default `~/kaggle-data/vesuvius`) holds everything derived from scroll data: partial CT mirrors, meshes, runs, logs and temp files. `VILLA_DIR` defaults to `$WINDEVAL_DATA/villa`. The checkpoint goes to `$WINDEVAL_DATA/hf/winding_model_9um/` (`.venv/bin/hf download scrollprize/winding_model_9um --local-dir "$WINDEVAL_DATA/hf/winding_model_9um"`).

## Data

Scroll data is CC BY-NC 4.0 and is never copied into this repository. Chunks stream on demand from the public bucket `https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com` into a local partial mirror:

```bash
PY=.venv/bin/python; B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com; D="$WINDEVAL_DATA"
$PY scripts/list_segments.py PHerc0139 > "$D/PHerc0139_mesh_listing.json"
$PY scripts/fetch_meshes.py "$D/PHerc0139_mesh_listing.json" "$D/data" on-20260102150214-2.399um.tifxyz
$PY scripts/make_mirror.py $B/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr \
    "$D/zarr/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr" 2
$PY scripts/link_segments.py "$D/data/PHerc0139/segments" on-20260102150214-2.399um.tifxyz "$D/eval/PHerc0139/segments_on_20260102150214"
$PY scripts/scale_segments.py "$D/eval/PHerc0139/segments_on_20260102150214" "$D/eval/PHerc0139/segments_on_20260102150214_ds4" 4
```

The released config reads PHerc0139 at `volume_scale: 2` with `segment_downscale: 4`, so meshes must be in level-2 voxel units; `scale_segments.py` divides the published full-resolution meshes by 4 (or use `patches/0002`). REPORT.md lists the exact volumes and mesh sets for every condition.

## Run

```bash
.venv/bin/python -m windeval.evaluate configs/gate_pherc0139_2399_l2.json --num-slabs 256 --num-workers 8
.venv/bin/python -m windeval.report "$WINDEVAL_DATA/runs/gate_pherc0139_2399_l2_z3904"
.venv/bin/python -m windeval.fit_transform fits/pherc0139_2399l2_to_9362.json
.venv/bin/python -m windeval.evaluate configs/cond2fair_pherc0139_9362_default.json --num-slabs 256 --num-workers 8
.venv/bin/python -m windeval.compare --ref "$WINDEVAL_DATA/runs/gate_pherc0139_2399_l2_z3904" \
    "$WINDEVAL_DATA/runs/cond2fair_pherc0139_9362_default" --band 40 49
```

Long runs can be queued with `scripts/run_queue.sh QUEUE_FILE`. A 256-slab run takes 9 to 12 minutes on the M4 Max.

## License

MIT, see [LICENSE](LICENSE). The scroll data used by the configs is CC BY-NC 4.0 and is not part of this repository.
