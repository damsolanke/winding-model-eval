# villa patches for running the winding-model evaluation path on macOS

Three independent diffs against villa `e0bbb8b40a2db58b1d71864f286eb85717e59e64`. Each applies on its own (`git apply --check` passes for each against a clean checkout) and keeps the default behaviour unchanged. Every file starts with its rationale; `git apply` skips that text.

| Patch | Files | What it does | Why |
|---|---|---|---|
| `0001-winding-dataset-spawn-workers.patch` | `winding_model_dataset.py`, `volume_slab_extractor.py`, `train_winding_model.py` | Segments rebuild their `GridRaycaster` after unpickling, the slab extractor drops per-process volume handles when pickled, and the trainer reads `dataloader_start_method` (default `"fork"`) | Forked DataLoader workers can abort in the Objective-C runtime on macOS; `"spawn"` needs a picklable dataset |
| `0002-segment-coordinate-divisor.patch` | `winding_model_dataset.py` | Optional per-dataset `segment_coordinate_divisor` divides mesh vertex coordinates on load | The published meshes are in full-resolution voxels while the released config samples level-2 slabs at one segment unit per sample; read as-is, precision drops from 0.44 to 0.13 |
| `0003-numpy-volume-fallback.patch` | new `numpy_volume.py`, `volume_slab_extractor.py` | If the volume-cartographer bindings are missing, `vc.Volume` falls back to a zarr-plus-NumPy sampler with the C++ sampler's semantics | The bindings need `vc_core` and do not build on macOS yet (PR #1975); `vc.grid_raycast` alone builds from its single source file (see `../native/`) |

Validation (`scripts/check_villa_patches.py`): 8 PHerc0139 slabs drawn through a patched checkout (villa's own dataset pickled into spawn workers, the published full-resolution meshes with `segment_coordinate_divisor: 4`, only the standalone `vc.grid_raycast` installed) against the windeval harness that produced every reported number. Labels identical; all 50,331,648 image voxels identical.

Example config for the patched path:

```json
{"volume_path": ".../PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr",
 "volume_id": "20260102150214", "volume_scale": 2,
 "segment_volume_id": "20260102150214", "segment_downscale": 4,
 "segment_coordinate_divisor": 4,
 "segments_path": "<folder of the published -on-20260102150214-2.399um meshes, one per winding name>"}
```

plus `"dataloader_start_method": "spawn"` at the top level of the training config.
