"""Score a winding-model checkpoint with villa's own dataset, losses and metrics.

The slabs, targets, losses and metric sums come from villa unchanged:
``WindingModelDataset`` (fresh procedural draws, as ``validation_mode:
"separate"`` uses them), ``compute_losses`` and ``accumulate_winding_metrics`` /
``finalize_winding_metrics`` from ``train_winding_model.py``. This module only
adds what the trainer does not have: a spatial restriction on where rays start,
per-slab bookkeeping (for bootstrap intervals), a per-winding label agreement
diagnostic, device selection, timings and I/O accounting. With a `replay`
block (windeval/replay.py) the slabs and labels come from the reference scan
and only the CT is re-sampled from another scan.

Usage: python -m windeval.evaluate CONFIG.json [--num-slabs N] [--out DIR]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from windeval import env

env.setup()

from windeval.lazy import LazyWindingDataset, apply_transform_overrides  # noqa: E402
from windeval.sampling import restrict_sources  # noqa: E402
from windeval.replay import ReplayDataset  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

import vc  # noqa: E402  (windeval/vcshim)
from vesuvius.neural_tracing.winding_models import winding_targets  # noqa: E402
from vesuvius.neural_tracing.winding_models.train_winding_model import (  # noqa: E402
    accumulate_winding_metrics,
    compute_losses,
    finalize_winding_metrics,
    seed_worker,
)
from vesuvius.neural_tracing.winding_models.winding_model import WindingModel  # noqa: E402
from vesuvius.neural_tracing.winding_models.winding_model_dataset import (  # noqa: E402
    WindingModelDataset,
)

SUM_KEYS = (
    "crossing_tp",
    "crossing_fp",
    "crossing_fn",
    "phase_abs_error",
    "phase_slabs",
    "count_error",
    "count_columns",
    "density_abs_error",
    "density_mean_sigma",
    "density_columns",
    "density_sigma_columns",
)


def empty_sums() -> dict:
    return {key: 0 if key.endswith(("_tp", "_fp", "_fn", "_slabs", "_columns")) else 0.0
            for key in SUM_KEYS}


def expand(path: str) -> str:
    return str(env.expand_path(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dir_bytes(path: Path) -> tuple[int, int]:
    total, files = 0, 0
    for root, _, names in os.walk(path):
        for name in names:
            if name.startswith(".") or name.endswith(".tmp"):
                continue
            total += os.path.getsize(os.path.join(root, name))
            files += 1
    return total, files


def slab_metadata(batch, slab: int, transform: np.ndarray) -> dict:
    origin_zyx = batch["ray_origin_zyx"][slab].numpy().astype(np.float64)
    direction_zyx = batch["ray_direction_zyx"][slab].numpy().astype(np.float64)
    origin_vol = origin_zyx[::-1] @ transform[:3, :3].T + transform[:3, 3]
    counts = batch["num_crossings"][slab].numpy()
    indices = batch["crossing_indices"][slab].numpy()
    supervised = counts >= 2
    columns = np.argwhere(supervised)
    windings = (np.concatenate([indices[i, j, : counts[i, j]] for i, j in columns])
                if len(columns) else np.zeros(0, dtype=np.int64))
    # Local winding spacing: ray samples per winding between consecutive labelled
    # crossings with consecutive indices (small = compressed, large = open).
    ts = batch["crossing_t"][slab].numpy()
    gaps = []
    for i, j in columns:
        t = ts[i, j, : counts[i, j]]
        idx = indices[i, j, : counts[i, j]]
        step = np.abs(np.diff(idx)) == 1
        gaps.extend(np.diff(t)[step].tolist())
    return {
        "origin_vol_zyx": [round(float(v), 2) for v in origin_vol[::-1]],
        "direction_zyx": [round(float(v), 4) for v in direction_zyx],
        "supervised_columns": int(supervised.sum()),
        "crossings": int(counts[supervised].sum()),
        "max_crossings_per_column": int(counts.max()) if counts.size else 0,
        "winding_span": [int(np.abs(windings).min()), int(np.abs(windings).max())] if windings.size else None,
        "median_winding_gap": float(np.median(gaps)) if gaps else None,
    }


def per_winding_agreement(one_out, one_batch, config, acc: dict) -> None:
    """Label-side agreement per winding index (diagnostic; pooled metrics stay villa's).

    For each labelled crossing, the distance to the nearest decoded phase
    passage, using the same per-slab offset registration and crossing_valid
    filter as accumulate_winding_metrics. acc[w] = [labels, within tolerance,
    sum of oriented residuals within 8 samples, count within 8 samples];
    oriented residual > 0 means the passage sits on the higher-winding side.
    """
    spacing = float(config.get("spacing", 1.0))
    tolerance = float(config.get("crossing_match_tolerance_wv", 2.0)) / spacing
    phase = one_out["phase"][0].numpy()
    target = one_batch["phase_target"][0].numpy()
    valid = one_batch["phase_valid"][0].numpy().astype(bool)
    crossing_valid = one_batch["crossing_valid"][0].numpy().astype(bool)
    ct = one_batch["crossing_t"][0].numpy()
    ci = one_batch["crossing_indices"][0].numpy()
    counts = one_batch["num_crossings"][0].numpy()
    offset = float(phase[valid].mean() - target[valid].mean()) if valid.any() else 0.0
    length = phase.shape[-1]
    for i, j in np.argwhere(counts >= 2):
        positions, _ = winding_targets.phase_passages(phase[i, j] - offset)
        peaks = np.clip(np.rint(positions).astype(np.int64), 0, length - 1)
        positions = np.sort(positions[crossing_valid[i, j][peaks]])
        n = counts[i, j]
        labels_t = ct[i, j, :n] / spacing
        labels_w = ci[i, j, :n]
        if len(positions):
            k = np.searchsorted(positions, labels_t)
            left = positions[np.clip(k - 1, 0, len(positions) - 1)] - labels_t
            right = positions[np.clip(k, 0, len(positions) - 1)] - labels_t
            d = np.where(np.abs(left) <= np.abs(right), left, right)
        else:
            d = np.full(n, np.inf)
        for w, dk in zip(labels_w.tolist(), d.tolist()):
            entry = acc.setdefault(abs(int(w)), [0, 0, 0.0, 0])
            entry[0] += 1
            if abs(dk) <= tolerance:
                entry[1] += 1
            if abs(dk) <= 8.0:
                entry[2] += dk if w > 0 else -dk
                entry[3] += 1


def bootstrap(records, n_boot=2000, seed=0) -> dict:
    """Percentile intervals by resampling slabs (each slab's sums move together)."""
    rng = np.random.default_rng(seed)
    table = np.array([[rec["sums"][key] for key in SUM_KEYS] for rec in records], dtype=np.float64)
    col = {key: i for i, key in enumerate(SUM_KEYS)}
    out = {"precision": [], "recall": [], "count_error": [], "phase_mae": []}
    for _ in range(n_boot):
        s = table[rng.integers(0, len(table), len(table))].sum(axis=0)
        tp, fp, fn = s[col["crossing_tp"]], s[col["crossing_fp"]], s[col["crossing_fn"]]
        out["precision"].append(tp / max(1.0, tp + fp))
        out["recall"].append(tp / max(1.0, tp + fn))
        out["count_error"].append(s[col["count_error"]] / max(1.0, s[col["count_columns"]]))
        out["phase_mae"].append(s[col["phase_abs_error"]] / max(1.0, s[col["phase_slabs"]]))
    return {key: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for key, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--num-slabs", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--weights", choices=("model", "model_ema"))
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--start-method", choices=("fork", "spawn"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    spec = json.loads(args.config.read_text())
    num_slabs = int(args.num_slabs or spec.get("num_slabs", 64))
    num_workers = int(args.num_workers if args.num_workers is not None else spec.get("num_workers", 4))
    weights_key = args.weights or spec.get("weights", "model")
    batch_size = int(spec.get("batch_size", 2))
    seed = int(spec.get("seed", 0))
    precision = args.precision or spec.get("precision", "fp32")
    run_name = (spec["name"] + (f"_{weights_key}" if weights_key != "model" else "")
                + (f"_{precision}" if precision != "fp32" else "") + args.name_suffix)
    out_dir = (args.out or Path(expand(spec.get("out_root", "${WINDEVAL_DATA}/runs"))) / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(expand(spec["checkpoint"]))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = copy.deepcopy(checkpoint["config"])
    config.update(spec.get("overrides", {}))
    config["datasets"] = [
        {**ds, "volume_path": expand(ds["volume_path"]), "segments_path": expand(ds["segments_path"])}
        for ds in spec["datasets"]
    ]
    config["num_samples"] = num_slabs
    config["batch_size"] = batch_size
    config["segment_cache_dir"] = expand(spec.get("segment_cache_dir", "${WINDEVAL_DATA}/cache/winding_segments"))
    if "segment_to_volume_scale" in spec:
        config["_segment_to_volume_scale"] = spec["segment_to_volume_scale"]
    config["_chunk_cache_bytes"] = int(spec.get("chunk_cache_bytes", 1 << 30))
    config["_io_threads"] = int(spec.get("io_threads", 8))
    replay = spec.get("replay")
    # Metrics and losses use the spacing the model actually read the slab at.
    metric_config = dict(config)
    if replay and "sample_spacing" in replay:
        metric_config["spacing"] = float(replay["sample_spacing"])
    vc.set_chunk_cache_budget(int(spec.get("chunk_cache_bytes", 1 << 30)))
    vc.set_chunk_cache_io_threads(int(spec.get("io_threads", 8)))

    random_seed = seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    t0 = time.time()
    dataset = WindingModelDataset(config)
    apply_transform_overrides(dataset, config)
    t_dataset = time.time() - t0
    region = spec.get("region_zyx")
    region_info = (restrict_sources(dataset, region["lo"], region["hi"], spec.get("source_windings"))
                   if region else None)
    n_segments = sum(len(v.segments) for v in dataset.volumes)
    print(f"dataset: {n_segments} segments in {t_dataset:.1f} s; region {region_info and {k: v for k, v in region_info.items() if k != 'origin_quads_by_winding'}}", flush=True)

    mirror_dirs = [Path(v.volume_path) for v in dataset.volumes]
    if replay:
        from vesuvius.neural_tracing.winding_models.volume_slab_extractor import VolumeSlabExtractor

        mirror_dirs.append(VolumeSlabExtractor.scaled_volume_path(
            env.expand_path(replay["volume_path"]), int(replay.get("volume_scale", 0))))
    mirror_before = [dir_bytes(p) for p in mirror_dirs]

    start_method = args.start_method or spec.get("start_method", "spawn")
    generator = torch.Generator()
    generator.manual_seed(seed + 1)  # the trainer seeds its validation loader with offset 1
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
        collate_fn=winding_targets.collate_winding_batch,
        pin_memory=False,
    )
    if num_workers > 0:
        loader_kwargs.update(multiprocessing_context=start_method, prefetch_factor=2, persistent_workers=False)
    if replay:
        loader_dataset = ReplayDataset(config, region, spec.get("source_windings"), replay)
    elif num_workers > 0 and start_method == "spawn":
        # Workers rebuild the dataset from the config (see windeval/lazy.py).
        loader_dataset = LazyWindingDataset(config, region, spec.get("source_windings"))
    else:
        loader_dataset = dataset
    loader = torch.utils.data.DataLoader(loader_dataset, **loader_kwargs)
    # Fork the workers before Metal is initialized in this process.
    iterator = iter(loader)

    device = env.pick_device(args.device or spec.get("device", "auto"))
    model = WindingModel(config.get("model"))
    model.load_state_dict(checkpoint[weights_key], strict=True)
    model.to(device).eval()
    if model.column_stride != dataset.column_stride:
        raise ValueError("model and dataset column strides differ")
    del checkpoint

    totals = empty_sums()
    by_winding: dict[int, list] = {}
    loss_sums: dict[str, float] = {}
    records = []
    timings = {"wait": [], "forward": [], "metrics": []}
    peak = {"mps_driver": 0, "mps_current": 0, "rss_main": 0, "rss_workers": 0}
    proc = psutil.Process()
    n_batches = 0
    t_start = time.time()
    t_mark = time.time()
    while True:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        t_got = time.time()
        timings["wait"].append(t_got - t_mark)
        dev_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            # bf16 mirrors the trainer's accelerate autocast (mixed_precision="bf16").
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
                output = model(dev_batch["slab_image"], dev_batch["slab_valid"])
            loss, loss_metrics = compute_losses(output, dev_batch, metric_config)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd = time.time()
            timings["forward"].append(t_fwd - t_got)
            loss_sums["val_loss"] = loss_sums.get("val_loss", 0.0) + float(loss)
            for key, value in loss_metrics.items():
                loss_sums[f"val_{key}"] = loss_sums.get(f"val_{key}", 0.0) + float(value)
            cpu_out = {k: v.detach().float().cpu() for k, v in output.items()}
        for slab in range(cpu_out["phase"].shape[0]):
            one_out = {k: v[slab: slab + 1] for k, v in cpu_out.items()}
            one_batch = {k: (v[slab: slab + 1] if torch.is_tensor(v) and v.dim() > 0 else v)
                         for k, v in batch.items()}
            sums = empty_sums()
            accumulate_winding_metrics(sums, one_out, one_batch, metric_config)
            per_winding_agreement(one_out, one_batch, metric_config, by_winding)
            for key in SUM_KEYS:
                totals[key] += sums[key]
            meta = slab_metadata(batch, slab, dataset.volumes[int(batch["volume_idx"][slab])].segment_to_volume_xyz)
            if "replay_ok" in batch:
                meta["replay_ok"] = bool(batch["replay_ok"][slab])
            records.append({"slab": len(records), "batch": n_batches, "sums": sums, **meta})
        t_met = time.time()
        timings["metrics"].append(t_met - t_fwd)
        n_batches += 1
        if device.type == "mps":
            peak["mps_driver"] = max(peak["mps_driver"], torch.mps.driver_allocated_memory())
            peak["mps_current"] = max(peak["mps_current"], torch.mps.current_allocated_memory())
        peak["rss_main"] = max(peak["rss_main"], proc.memory_info().rss)
        try:
            peak["rss_workers"] = max(peak["rss_workers"], sum(c.memory_info().rss for c in proc.children(recursive=True)))
        except psutil.Error:
            pass
        if n_batches % 8 == 0 or n_batches == 1:
            running = finalize_winding_metrics(dict(totals))
            print(f"[{n_batches * batch_size}/{num_slabs}] {time.time() - t_start:.0f}s "
                  f"fwd {np.mean(timings['forward']):.2f}s wait {np.mean(timings['wait']):.2f}s "
                  + " ".join(f"{k}={v:.4f}" for k, v in running.items()), flush=True)
        t_mark = time.time()

    wall = time.time() - t_start
    metrics = finalize_winding_metrics(dict(totals))
    losses = {k: v / max(1, n_batches) for k, v in loss_sums.items()}
    mirror_after = [dir_bytes(p) for p in mirror_dirs]
    streamed = sum(a[0] - b[0] for a, b in zip(mirror_after, mirror_before))
    chunk_files = sum(a[1] for a in mirror_after)

    def stats(values):
        values = np.asarray(values[1:] if len(values) > 2 else values, dtype=np.float64)
        return {"mean": float(values.mean()), "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)), "min": float(values.min()), "max": float(values.max())}

    result = {
        "name": run_name,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256(ckpt_path),
        "weights": weights_key,
        "villa_commit": git_head(env.VILLA_DIR),
        "device": str(device),
        "torch": torch.__version__,
        "precision": precision,
        "num_slabs": len(records),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "start_method": start_method,
        "seed": seed,
        "segments": n_segments,
        "region": region,
        "region_info": region_info,
        "metrics": metrics,
        "losses": losses,
        "totals": totals,
        "bootstrap_95": bootstrap(records) if len(records) > 1 else None,
        "label_agreement_by_winding": {
            str(w): {"labels": e[0], "within_tolerance": e[1],
                     "frac_within_tolerance": e[1] / max(1, e[0]),
                     "mean_oriented_residual_within_8": e[2] / max(1, e[3])}
            for w, e in sorted(by_winding.items())
        },
        "supervised_columns": int(sum(r["supervised_columns"] for r in records)),
        "crossings_labelled": int(sum(r["crossings"] for r in records)),
        "timing_s": {
            "wall": wall,
            "per_slab_wall": wall / max(1, len(records)),
            "dataset_build": t_dataset,
            "forward_per_batch": stats(timings["forward"]),
            "data_wait_per_batch": stats(timings["wait"]),
            "metrics_per_batch": stats(timings["metrics"]),
        },
        "memory_bytes": peak,
        "data": {
            "mirror_dirs": [str(p) for p in mirror_dirs],
            "bytes_streamed_this_run": streamed,
            "mirror_bytes_total": sum(a[0] for a in mirror_after),
            "mirror_chunk_files": chunk_files,
        },
        "config_overrides": spec.get("overrides", {}),
        "replay": replay,
        "replay_failed_slabs": sum(1 for r in records if r.get("replay_ok") is False),
        "segment_to_volume_scale": spec.get("segment_to_volume_scale"),
        "datasets": config["datasets"],
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    with open(out_dir / "slabs.jsonl", "w") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
    print(json.dumps({k: result[k] for k in ("metrics", "bootstrap_95", "num_slabs", "supervised_columns", "crossings_labelled")}, indent=1))
    print(json.dumps({k: result[k] for k in ("timing_s", "memory_bytes", "data")}, indent=1))


if __name__ == "__main__":
    main()
