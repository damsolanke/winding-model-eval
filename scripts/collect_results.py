"""Copy run outputs (result.json, slabs.jsonl) into results/runs/ with local paths made portable.

Absolute paths under the data root and the home directory are rewritten to
${WINDEVAL_DATA} and ~ so the committed results carry no machine-specific paths.

Usage: python scripts/collect_results.py RUN_NAME [RUN_NAME ...]   (names under $WINDEVAL_DATA/runs)
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WINDEVAL_DATA", "~/kaggle-data/vesuvius")).expanduser()


def portable(text: str) -> str:
    text = text.replace(str(DATA), "${WINDEVAL_DATA}")
    text = text.replace(str(ROOT), ".")
    return text.replace(str(Path.home()), "~")


def main():
    for name in sys.argv[1:]:
        src = DATA / "runs" / name
        dst = ROOT / "results" / "runs" / name
        dst.mkdir(parents=True, exist_ok=True)
        for file in ("result.json", "slabs.jsonl"):
            (dst / file).write_text(portable((src / file).read_text()))
        print("collected", name)


if __name__ == "__main__":
    main()
