"""Create a local partial mirror of one level of a remote OME-Zarr volume (metadata only).

Usage: make_mirror.py REMOTE_ZARR_URL LOCAL_ZARR_DIR LEVEL [LEVEL ...]
Chunks are fetched lazily by the vc shim; this writes group metadata, the per-level .zarray,
any transform.json / metadata.json, and the .windeval_remote pointer per level.
"""
import sys
import urllib.error
import urllib.request
from pathlib import Path


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def main():
    remote = sys.argv[1].rstrip("/")
    local = Path(sys.argv[2])
    levels = sys.argv[3:]
    local.mkdir(parents=True, exist_ok=True)
    for name in (".zattrs", ".zgroup", "transform.json", "metadata.json"):
        data = get(f"{remote}/{name}")
        if data is not None:
            (local / name).write_bytes(data)
            print("wrote", name)
    for level in levels:
        data = get(f"{remote}/{level}/.zarray")
        if data is None:
            raise SystemExit(f"no .zarray at level {level}")
        (local / level).mkdir(exist_ok=True)
        (local / level / ".zarray").write_bytes(data)
        (local / level / ".windeval_remote").write_text(f"{remote}/{level}\n")
        print("level", level, data.decode().replace("\n", " ")[:200])


if __name__ == "__main__":
    main()
