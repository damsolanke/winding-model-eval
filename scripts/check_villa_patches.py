"""Check the villa patches in patches/ against the harness on real slabs.

Run twice, then compare:
  mode "harness": unpatched villa + windeval's vc shim + level-2 mesh copies, spawn
                  workers through windeval.lazy (the setup every reported run used)
  mode "patched": patched villa checkout (VILLA_DIR) + only the standalone vc.grid_raycast,
                  the published full-resolution meshes with segment_coordinate_divisor 4,
                  villa's own dataset object pickled into spawn workers
  mode "compare": labels must match exactly; images may differ only where the
                  C++ "absent chunk zeroes the sample" rule and zarr's fill value disagree

Usage:
  VILLA_DIR=<unpatched> python scripts/check_villa_patches.py harness OUT_A.npz
  VILLA_DIR=<patched> WINDEVAL_VC_PATH=<dir holding vc/grid_raycast*.so> \
      python scripts/check_villa_patches.py patched OUT_B.npz
  python scripts/check_villa_patches.py compare OUT_A.npz OUT_B.npz
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "gate_pherc0139_2399_l2.json"
LABEL_KEYS = ("phase_target", "phase_valid", "crossing_valid", "crossing_t", "num_crossings", "crossing_indices")
N_SAMPLES = 8


def load_config():
    sys.path.insert(0, str(ROOT))
    from windeval import env

    spec = json.loads(CONFIG.read_text())
    import torch

    ckpt = torch.load(env.expand_path(spec["checkpoint"]), map_location="cpu", weights_only=False)
    config = copy.deepcopy(ckpt["config"])
    config.update(spec["overrides"])
    config["datasets"] = [{**d, "volume_path": str(env.expand_path(d["volume_path"])),
                           "segments_path": str(env.expand_path(d["segments_path"]))} for d in spec["datasets"]]
    config["num_samples"] = N_SAMPLES
    config["segment_cache_dir"] = str(env.expand_path("${WINDEVAL_DATA}/cache/winding_segments"))
    return spec, config


def draw(dataset, collate):
    import torch

    from vesuvius.neural_tracing.winding_models.train_winding_model import seed_worker

    generator = torch.Generator()
    generator.manual_seed(1001)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2,
                                         worker_init_fn=seed_worker, generator=generator,
                                         collate_fn=collate, multiprocessing_context="spawn")
    out = {}
    for i, batch in enumerate(loader):
        for key in ("slab_image",) + LABEL_KEYS:
            out[f"{i}_{key}"] = batch[key][0].numpy()
    return out


def main():
    import numpy as np

    mode = sys.argv[1]
    if mode == "compare":
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
        report = {"samples": N_SAMPLES, "labels_identical": True, "image_voxels_differing": 0, "image_voxels": 0}
        for i in range(N_SAMPLES):
            for key in LABEL_KEYS:
                x, y = a[f"{i}_{key}"], b[f"{i}_{key}"]
                if not np.array_equal(x, y, equal_nan=x.dtype.kind == "f"):
                    report["labels_identical"] = False
                    report.setdefault("label_mismatch", []).append(f"{i}:{key}")
            d = a[f"{i}_slab_image"].astype(int) - b[f"{i}_slab_image"].astype(int)
            report["image_voxels_differing"] += int((d != 0).sum())
            report["image_voxels"] += int(d.size)
            report.setdefault("max_abs_image_diff", 0)
            report["max_abs_image_diff"] = max(report["max_abs_image_diff"], int(np.abs(d).max()))
        print(json.dumps(report, indent=1))
        return

    spec, config = load_config()
    out_path = Path(sys.argv[2])
    if mode == "harness":
        from windeval import env

        env.setup()
        from vesuvius.neural_tracing.winding_models import winding_targets
        from windeval.lazy import LazyWindingDataset

        dataset = LazyWindingDataset(config, spec["region_zyx"])
    elif mode == "patched":
        sys.path.insert(0, os.environ["WINDEVAL_VC_PATH"])
        sys.path.insert(0, str(Path(os.environ["VILLA_DIR"]).expanduser() / "vesuvius" / "src"))
        from windeval import env

        os.environ.setdefault("TMPDIR", str(env.DATA_ROOT / "tmp"))
        import vc

        assert not hasattr(vc, "Volume"), "patched mode must run without the vc shim"
        from vesuvius.neural_tracing.winding_models import winding_targets
        from vesuvius.neural_tracing.winding_models.winding_model_dataset import WindingModelDataset
        from windeval.sampling import restrict_sources

        config["datasets"] = [{**d, "segments_path": str(env.expand_path(
            "${WINDEVAL_DATA}/eval/PHerc0139/segments_on_20260102150214")), "segment_coordinate_divisor": 4}
            for d in config["datasets"]]
        dataset = WindingModelDataset(config)
        restrict_sources(dataset, spec["region_zyx"]["lo"], spec["region_zyx"]["hi"])
    else:
        raise SystemExit(f"unknown mode {mode}")
    np.savez_compressed(out_path, **draw(dataset, winding_targets.collate_winding_batch))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
