"""Download selected tifxyz mesh variants for one sample from the open-data bucket.

Usage: fetch_meshes.py LISTING_JSON DEST_ROOT PATTERN [PATTERN ...]
PATTERN is a substring matched against the mesh sub-path (e.g. 'intermediate/tifxyz_original',
'-on-20250728140407-9.362um.tifxyz'). Files already present with the right size are skipped.
Appends one line per run to DEST_ROOT/download_ledger.tsv (bytes fetched).
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"


def fetch(job):
    key, size, dest = job
    if dest.is_file() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(f"{BUCKET}/{key}") as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"size mismatch for {key}")
    tmp.replace(dest)
    return size


def main():
    listing = json.load(open(sys.argv[1]))
    root = Path(sys.argv[2])
    patterns = sys.argv[3:]
    jobs = []
    for seg, files in listing.items():
        for rel, size in files:
            if any(p in rel for p in patterns):
                jobs.append((seg + rel, size, root / (seg + rel)))
    t0 = time.time()
    with ThreadPoolExecutor(16) as pool:
        fetched = sum(pool.map(fetch, jobs))
    dt = time.time() - t0
    with open(root / "download_ledger.tsv", "a") as ledger:
        ledger.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\tmeshes\t{' '.join(patterns)}\t{len(jobs)} files\t{fetched} bytes\t{dt:.1f} s\n")
    print(f"{len(jobs)} files, fetched {fetched/1e6:.1f} MB in {dt:.1f} s")


if __name__ == "__main__":
    main()
