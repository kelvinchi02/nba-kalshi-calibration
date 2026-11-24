"""Run every notebook in notebooks/ top to bottom and save it with its outputs.

Each notebook is executed with nbconvert in a fresh kernel, with its own folder as the working
directory. If any cell raises, the script prints the error and exits with code 1. A notebook
that fails is left as it was on disk.

Usage:
    python scripts/run_notebooks.py                    # all notebooks, in name order
    python scripts/run_notebooks.py 02_calibration     # only the named ones
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
CELL_TIMEOUT_S = 3600


def select(names: list[str]) -> list[Path]:
    """All notebooks in name order, or only the named ones (with or without .ipynb)."""
    if not names:
        return sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    paths = [NOTEBOOK_DIR / (n if n.endswith(".ipynb") else n + ".ipynb") for n in names]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"Not found in {NOTEBOOK_DIR}: {', '.join(missing)}")
    return paths


def run(path: Path) -> None:
    """Execute one notebook and write it back with its outputs."""
    nb = nbformat.read(path, as_version=4)
    # On Windows the kernel client falls back to a selector thread and says so. It works.
    warnings.filterwarnings("ignore", message="Proactor event loop does not implement")
    # record_timing=False keeps timestamps out of the file, so a rerun gives the same bytes.
    # --log-level=ERROR keeps the kernel's start-up warnings out of the console.
    # coalesce_streams joins the pieces of a cell's printed output, which otherwise arrive
    # in one or two chunks depending on timing.
    executor = ExecutePreprocessor(timeout=CELL_TIMEOUT_S, kernel_name="python3",
                                   record_timing=False, coalesce_streams=True,
                                   extra_arguments=["--log-level=ERROR"])
    executor.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    nbformat.write(nb, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="notebooks to run (default: all)")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")  # inherited by the kernels

    for path in select(args.names):
        start = time.time()
        print(f"{path.name} ... ", end="", flush=True)
        try:
            run(path)
        except CellExecutionError as exc:
            print("FAILED")
            print(exc, file=sys.stderr)
            raise SystemExit(1)
        print(f"ok ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
