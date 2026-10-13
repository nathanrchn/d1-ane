"""Specialize a build's packages for this Mac's Neural Engine ahead of serving, and report where they were placed.

    d1ane compile [--build build/d1-omni-600M] [--force]

A package is compiled on its first load and cached per macOS build and interpreter, so run this with the Python
that will serve. Later loads take well under a second. --force drops this build's cached specializations first.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from . import BUILD
from .placement import audit, cache_dir, inspect


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--force", action="store_true", help="drop cached specializations of this build first")
    a = ap.parse_args(argv)
    from coreai.runtime import AIModel, ComputeUnitKind, SpecializationOptions

    man = json.loads((a.build / "manifest.json").read_text())
    files = [s["file"] for s in man["stages"]] + [man[k]["file"] for k in ("vision", "audio") if k in man]
    spec = SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())
    record = a.build / "compile.json"
    loop, times = asyncio.new_event_loop(), json.loads(record.read_text())["seconds"] if record.is_file() else {}
    for f in files:
        if a.force:
            shutil.rmtree(cache_dir(a.build / f), ignore_errors=True)
        cold = not any("ane" in spec_["preferred"] for spec_ in inspect(a.build / f, []))
        t0 = time.time()
        loop.run_until_complete(AIModel.load(a.build / f, specialization_options=spec))
        dt = round(time.time() - t0, 1)
        if cold or f not in times:          # keep the cold compile time of a package that was already cached
            times[f] = dt
        print(f"{f}: {dt}s ({'compiled' if cold else 'cached'})", flush=True)
    report = audit(a.build)
    for f, st in report["stages"].items():
        print(f"{f}: {st['status']} (bonded mode {st['modes']}, {len(st['entries'])} entries, gpu regions {st['gpu_regions']})")
    record.write_text(json.dumps({"seconds": times, "placement": report}, indent=1))
    return int(report["status"] != "fully_ane")


if __name__ == "__main__":
    sys.exit(main())
