"""Where the compiled entries of a build were placed, read from the Core AI specialization cache (no model load).

    python -m d1ane.placement --build build/d1-omni-600M [--strict]

A package is specialized on its first load and cached under ~/Library/Caches/coreai-cache/<macOS build>/<executable>/
<package hash>/. Each specialization's MPSGraph manifest records the preferred device, the ANE bonded compile mode
and, per entry, whether it was placed fully on the ANE with no GPU activity. This reads cached specializations; it
does not prove live hardware execution.
"""
from __future__ import annotations

import argparse
import json
import plistlib
import re
import subprocess
import sys
from pathlib import Path

CACHE = Path.home() / "Library/Caches/coreai-cache"
PREFERRED = {1: "gpu", 2: "ane", 0: "cpu"}


def os_build() -> str:
    return subprocess.run(["sw_vers", "-buildVersion"], check=True, capture_output=True, text=True).stdout.strip()


def cache_dir(package: Path, executable: str = sys.executable) -> Path:
    return CACHE / os_build() / Path(executable).name.replace("_", "-") / (package / "main.hash").read_bytes().hex()


def inspect(package: Path, entries: list[str], executable: str = sys.executable) -> list[dict]:
    """One record per cached specialization of the package."""
    out = []
    for mf in sorted(cache_dir(package, executable).glob("*/model.aimodelx/**/manifest.plist")):
        if ".mpsgraphpackage" not in str(mf):
            continue
        raw = mf.read_bytes()
        pref = {int(x) for x in re.findall(rb'preferredDevice\\?"\s*:\s*(\d+)', raw)}
        rec = {"preferred": sorted(PREFERRED.get(p, str(p)) for p in pref),
               "modes": sorted({int(x) for x in re.findall(rb"aneBondedCompileMode\W+(\d+)", raw)}),
               "entries": {}, "ane_regions": 0, "gpu_regions": 0}
        for fields in plistlib.loads(raw).get("Package Version", {}).values():
            for module in fields.get("Optimized Modules", {}).values():
                graph = mf.parent / module.get("File Name", "")
                if graph.is_file():
                    blob = graph.read_bytes()
                    rec["ane_regions"] += len(set(re.findall(rb"[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+", blob)))
                    rec["gpu_regions"] += len(set(re.findall(rb"[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+", blob)))
                for sym, attrs in module.get("Entry Function Attributes", {}).items():
                    entry = next((e for e in entries if sym == e or sym.startswith(e + "_")), None)
                    if entry is None:
                        continue
                    flags = {a for a in attrs if isinstance(a, str)}
                    full = "mps.fullyPlacedOnANE" in flags and "mps.noGPUActivity" in flags
                    rec["entries"][entry] = "fully_ane" if full else "not_fully_ane"
        out.append(rec)
    return out


def audit(build: Path, executable: str = sys.executable) -> dict:
    """Status of the ANE-preferred specialization of every stage: fully_ane | partial | missing."""
    man = json.loads((build / "manifest.json").read_text())
    report = {"os_build": os_build(), "stages": {}}
    for st in man["stages"] + [man[k] for k in ("vision", "audio") if k in man]:
        entries = list(st["entries"].values()) if isinstance(st["entries"], dict) else list(st["entries"])
        specs = [s for s in inspect(build / st["file"], entries, executable) if "ane" in s["preferred"]]
        placed = {e: next((s["entries"][e] for s in specs if e in s["entries"]), "missing") for e in entries}
        report["stages"][st["file"]] = {
            "entries": placed, "modes": sorted({m for s in specs for m in s["modes"]}),
            "gpu_regions": sum(s["gpu_regions"] for s in specs), "ane_regions": sum(s["ane_regions"] for s in specs),
            "status": "fully_ane" if specs and all(v == "fully_ane" for v in placed.values())
                      and not any(s["gpu_regions"] for s in specs) else "missing" if not specs else "partial"}
    report["status"] = "fully_ane" if all(s["status"] == "fully_ane" for s in report["stages"].values()) else "partial"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--executable", default=sys.executable, help="the interpreter that compiled (cache key)")
    ap.add_argument("--strict", action="store_true", help="exit 1 unless every stage is fully_ane")
    a = ap.parse_args(argv)
    report = audit(a.build, a.executable)
    print(json.dumps(report, indent=1))
    return int(a.strict and report["status"] != "fully_ane")


if __name__ == "__main__":
    sys.exit(main())
