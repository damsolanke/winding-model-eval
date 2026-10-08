"""Ray-origin restriction for WindingModelDataset (no import side effects)."""

from __future__ import annotations

import numpy as np
import torch


def restrict_sources(dataset, lo_zyx, hi_zyx, source_windings=None) -> dict:
    """Keep only ray-origin quads whose sampled point lies inside the region.

    The region is in voxels of each dataset's selected volume array. Every
    segment stays loaded, so bracket windings and rasterized targets are
    unchanged; only where rays start is limited. The sampling CDF is rebuilt
    with the dataset's own rule (sampling_weight x number of origin quads).
    """
    lo = np.asarray(lo_zyx, dtype=np.float64)
    hi = np.asarray(hi_zyx, dtype=np.float64)
    kept_total, before_total = 0, 0
    per_segment = {}
    for volume in dataset.volumes:
        transform = volume.segment_to_volume_xyz
        for segment in volume.segments:
            cells = segment.sample_cells
            before_total += len(cells)
            if not len(cells):
                continue
            rows, cols = cells[:, 0], cells[:, 1]
            # _point_and_normal draws the point (quad[1, 0] + quad[0, 1]) / 2.
            points = (segment.xyz[rows + 1, cols].astype(np.float64)
                      + segment.xyz[rows, cols + 1].astype(np.float64)) / 2.0
            zyx = (points @ transform[:3, :3].T + transform[:3, 3])[:, ::-1]
            keep = ((zyx >= lo) & (zyx < hi)).all(axis=1)
            if source_windings is not None and segment.winding_idx not in source_windings:
                keep[:] = False
            segment.sample_cells = cells[keep]
            kept_total += int(keep.sum())
            per_segment[segment.winding_idx] = per_segment.get(segment.winding_idx, 0) + int(keep.sum())
    weights = [
        dataset.volumes[volume_idx].sampling_weight * len(segment.sample_cells)
        for volume_idx, segment in dataset._segment_lookup
    ]
    weights = torch.tensor(weights, dtype=torch.float64)
    if weights.sum() <= 0:
        raise ValueError("region leaves no ray-origin quads")
    dataset._segment_cdf = torch.cumsum(weights, dim=0)
    return {"origin_quads_before": before_total, "origin_quads_kept": kept_total,
            "origin_quads_by_winding": per_segment}
