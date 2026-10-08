"""A dataset wrapper that builds villa's WindingModelDataset inside each DataLoader worker.

Forked workers on macOS intermittently crash (Objective-C +initialize fork-safety
abort, then segfaults with the check disabled). With the "spawn" start method
the dataset must be picklable, and WindingModelDataset holds nanobind
GridRaycaster objects that are not, so each worker builds its own copy from the
config on first use. Sampling only uses the worker's torch RNG, which the
DataLoader seeds per worker, so a spawn run draws the same slabs as a fork run
with the same seed and worker count.
"""

from __future__ import annotations

import numpy as np
import torch


def apply_transform_overrides(dataset, config: dict) -> None:
    """Optional non-integer segment-to-volume scale per dataset (`_segment_to_volume_scale`).

    villa derives the transform from integer `segment_downscale` / `volume_scale`;
    resampling a 7.91 um scan to the model's ~9.6 um needs a fractional factor.
    """
    scales = config.get("_segment_to_volume_scale")
    if scales is None:
        return
    if not isinstance(scales, (list, tuple)):
        scales = [scales] * len(dataset.volumes)
    for index, (volume, scale) in enumerate(zip(dataset.volumes, scales)):
        if scale is None:
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] *= float(scale)
        volume.segment_to_volume_xyz = matrix
        dataset.slab_extractor.segment_to_volume_xyz[index] = matrix


def build_dataset(config: dict, region: dict | None, source_windings=None):
    import vc

    from vesuvius.neural_tracing.winding_models.winding_model_dataset import WindingModelDataset
    from windeval.sampling import restrict_sources

    vc.set_chunk_cache_budget(int(config.get("_chunk_cache_bytes", 1 << 30)))
    vc.set_chunk_cache_io_threads(int(config.get("_io_threads", 8)))
    dataset = WindingModelDataset(config)
    apply_transform_overrides(dataset, config)
    if region:
        restrict_sources(dataset, region["lo"], region["hi"], source_windings)
    return dataset


class LazyWindingDataset(torch.utils.data.Dataset):
    def __init__(self, config: dict, region: dict | None, source_windings=None):
        self.config = config
        self.region = region
        self.source_windings = source_windings
        self.num_samples = int(config.get("num_samples", 500_000))
        self._dataset = None

    def __len__(self) -> int:
        return self.num_samples

    def build(self):
        if self._dataset is None:
            from windeval import env

            env.setup()
            self._dataset = build_dataset(self.config, self.region, self.source_windings)
        return self._dataset

    def __getitem__(self, index):
        return self.build()[index]

    def __getstate__(self):
        state = dict(self.__dict__)
        for key in list(state):
            if key.startswith("_") and key not in ("_dataset",) and not isinstance(state[key], (int, float, str, dict, list, tuple, type(None))):
                state[key] = None
        state["_dataset"] = None
        return state
