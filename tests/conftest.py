import os
from pathlib import Path


def pytest_configure(config):
    # Keep test scratch files next to the project data, not in the system temp dir.
    if not config.option.basetemp:
        base = Path(os.environ.get("WINDEVAL_DATA", "~/kaggle-data/vesuvius")).expanduser() / "tmp" / "pytest"
        config.option.basetemp = str(base)
