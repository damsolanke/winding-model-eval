"""Score another scan on exactly the rays and labels of a reference dataset.

Each sample is drawn by villa's WindingModelDataset on the reference volume
(rays, bracket windings, observability and rendered targets all come from
there). The slab image is then re-sampled from a second volume along the same
slab frame, through a fitted affine from the reference volume array to the
second volume's array. The only thing that changes is the CT the model sees.

Options (the `replay` block of an evaluate config):
  volume_path      second volume (OME-Zarr group or array dir)
  volume_scale     level to open (default 0)
  transform        JSON with `matrix_source_to_target_xyz` (reference array xyz -> second array xyz),
                   or "identity"
  sample_spacing   slab sample spacing in reference segment units (default: the dataset's spacing).
                   A different value means the model reads the second scan at another voxel size;
                   targets are then re-rendered for that frame with villa's own rasterizer, from
                   the same ray and bracket segments, and observability from the reference scan.
  blur_sigma       Gaussian blur of the slab image along the ray axis, in samples (default 0)
  zaxis_blur_sigma Gaussian blur along the second scan's z axis (its rotation axis), in that scan's
                   voxels: the slab is re-sampled at z offsets -2.5 sigma..+2.5 sigma (1-voxel steps)
                   and averaged with Gaussian weights (default 0)
"""

from __future__ import annotations

import json

import numpy as np
import torch

from windeval.lazy import LazyWindingDataset


def load_matrix(spec: str) -> np.ndarray:
    from windeval import env

    if spec == "identity":
        return np.eye(4)
    data = json.loads(env.expand_path(spec).read_text())
    matrix = np.eye(4)
    matrix[:3] = np.asarray(data["matrix_source_to_target_xyz"], dtype=np.float64)
    return matrix


class ReplayDataset(LazyWindingDataset):
    def __init__(self, config: dict, region, source_windings, replay: dict):
        super().__init__(config, region, source_windings)
        self.replay = dict(replay)
        self._alt = None
        self._ref_native = None
        self._spacing = None

    def build(self):
        dataset = super().build()
        if self._alt is not None:
            return dataset
        from windeval import env
        from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor

        original_extract = dataset.slab_extractor.extract

        def recording_extract(volume_idx, direction, ray_origin):
            dataset._windeval_last = (int(volume_idx), np.array(direction, dtype=np.float64),
                                      np.array(ray_origin, dtype=np.float64))
            return original_extract(volume_idx, direction, ray_origin)

        dataset.slab_extractor.extract = recording_extract
        original_targets = dataset._slab_targets

        def recording_targets(frame, raster_specs, slab_image, slab_valid, *, crossings_only=False):
            dataset._windeval_specs = raster_specs
            return original_targets(frame, raster_specs, slab_image, slab_valid, crossings_only=crossings_only)

        dataset._slab_targets = recording_targets

        if len(dataset.volumes) != 1:
            raise ValueError("replay supports one reference dataset")
        seg_to_ref = dataset.volumes[0].segment_to_volume_xyz
        seg_to_alt = load_matrix(self.replay.get("transform", "identity")) @ seg_to_ref
        alt_path = VolumeSlabExtractor.scaled_volume_path(
            env.expand_path(self.replay["volume_path"]), int(self.replay.get("volume_scale", 0)))
        self._spacing = float(self.replay.get("sample_spacing", dataset.spacing))
        common = dict(transverse_size=dataset.transverse_size, ray_length=dataset.ray_length,
                      spacing=self._spacing, sampling=str(self.config.get("sampling", "trilinear")),
                      tile_size=int(self.config.get("tile_size", 32)))
        self._alt = VolumeSlabExtractor([alt_path], segment_to_volume_xyz=[seg_to_alt], **common)
        # one voxel along the second scan's z axis, expressed in segment coordinates
        self._z_step_segment = np.linalg.solve(seg_to_alt[:3, :3], np.array([0.0, 0.0, 1.0]))
        if self._spacing != dataset.spacing:
            self._ref_native = VolumeSlabExtractor([dataset.volumes[0].volume_path],
                                                   segment_to_volume_xyz=[seg_to_ref], **common)
        return dataset

    def __getitem__(self, index):
        dataset = self.build()
        sample = dataset[index]
        _, direction, origin = dataset._windeval_last
        sample["replay_ok"] = torch.tensor(True)
        if self._ref_native is None:
            image, valid, frame = self._alt.extract(0, direction, origin)
            z_sigma = float(self.replay.get("zaxis_blur_sigma", 0.0))
            if z_sigma > 0:
                reach = int(np.ceil(2.5 * z_sigma))
                offsets = np.arange(-reach, reach + 1)
                weights = np.exp(-0.5 * (offsets / z_sigma) ** 2)
                weights /= weights.sum()
                acc = np.zeros(image.shape, dtype=np.float32)
                for k, w in zip(offsets, weights):
                    shifted = image if k == 0 else self._alt.extract(0, direction, origin + k * self._z_step_segment)[0]
                    acc += np.float32(w) * shifted.astype(np.float32)
                image = acc
        else:
            ref_image, ref_valid, frame = self._ref_native.extract(0, direction, origin)
            saved = dataset.spacing
            dataset.spacing = self._spacing
            try:
                targets = dataset._slab_targets(frame, dataset._windeval_specs, ref_image, ref_valid)
            finally:
                dataset.spacing = saved
            if targets is None:
                sample["replay_ok"] = torch.tensor(False)
                for key in ("phase_valid", "crossing_valid"):
                    sample[key] = torch.zeros_like(sample[key])
                sample["num_crossings"] = torch.zeros_like(sample["num_crossings"])
            else:
                sample.update(targets)
            image, valid, _ = self._alt.extract(0, direction, origin)
            sample["spacing"] = torch.tensor(self._spacing, dtype=torch.float32)
            sample["ray_extent"] = torch.tensor((dataset.ray_length - 1) * self._spacing, dtype=torch.float32)
            sample["slab_origin_zyx"] = torch.from_numpy(frame.origin[::-1].copy()).float()
            sample["slab_axis_a_zyx"] = torch.from_numpy(frame.axis_a[::-1].copy()).float()
            sample["slab_axis_b_zyx"] = torch.from_numpy(frame.axis_b[::-1].copy()).float()
        sigma = float(self.replay.get("blur_sigma", 0.0))
        if sigma > 0:
            import scipy.ndimage

            image = scipy.ndimage.gaussian_filter1d(image.astype(np.float32), sigma, axis=-1, mode="nearest")
        sample["slab_image"] = torch.from_numpy(np.ascontiguousarray(image))
        sample["slab_valid"] = torch.from_numpy(np.ascontiguousarray(valid)).bool()
        return sample
