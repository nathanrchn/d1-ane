"""Tabulate bench/results/*.json as Markdown (stdout).

    python bench/report.py [--results bench/results] [--build build/d1-omni-600M-fp16] > RESULTS.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"coreai-ane": "Core AI, ANE", "coreai-ane-int8": "Core AI, ANE, int8 weights", "coreai-gpu": "Core AI, GPU",
         "torch-mps-fp16": "PyTorch MPS fp16", "torch-mps-fp32": "PyTorch MPS fp32"}
ORDER = list(NAMES)
REF = "torch-mps-fp32"
VERSUS = ("coreai-gpu", "torch-mps-fp16")
PARITY_HEAD = ["Decisions", "Accuracy", "Argmax ≠ FP32", "mean max\\|Δp\\|", "p99 max\\|Δp\\|", "mean KL", "Median ms",
               "p95 ms", "Decisions/s"]


def load(results: Path, task: str) -> dict:
    out = {}
    for e in ORDER:
        f = results / f"{task}_{e}.json"
        if f.is_file():
            out[e] = json.loads(f.read_text())
    return out


def table(head: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(head) - 1)
    line = "| " + " | ".join(head) + " |\n| " + " | ".join("---" if a == "l" else "---:" for a in align) + " |\n"
    return line + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + "\n"


def ms(v: float | None) -> str:
    return "–" if v is None else f"{v:.1f}" if v < 100 else f"{v:.0f}"


def parity(rows: list[dict], ref: dict, is_ref: bool, per_s: float | None = None) -> list[str]:
    """PARITY_HEAD cells of one engine's rows against the FP32 reference rows (by id)."""
    ok = [r for r in rows if "error" not in r and r["id"] in ref]
    t = np.array([r["ms"] for r in ok])
    dp = np.array([np.abs(np.array(r["probs"]) - np.array(ref[r["id"]]["probs"])).max() for r in ok])
    kl = np.array([float((q * (np.log(q + 1e-12) - np.log(np.array(r["probs"]) + 1e-12))).sum())
                   for r in ok for q in [np.array(ref[r["id"]]["probs"])]])
    flips = sum(r["pred"] != ref[r["id"]]["pred"] for r in ok)
    return [len(ok), f"{100 * np.mean([r['pred'] == r['gold'] for r in ok]):.2f}%",
            "reference" if is_ref else f"{flips} ({100 * flips / len(ok):.2f}%)",
            "–" if is_ref else f"{dp.mean():.1e}", "–" if is_ref else f"{np.percentile(dp, 99):.1e}",
            "–" if is_ref else f"{kl.mean():.1e}", f"{np.median(t):.1f}", f"{np.percentile(t, 95):.1f}",
            f"{per_s if per_s is not None else 1e3 / t.mean():.0f}"]


def build_section(build: Path) -> str:
    man = json.loads((build / "manifest.json").read_text())
    comp = json.loads((build / "compile.json").read_text()) if (build / "compile.json").is_file() else {}
    rows = []
    for st in man["stages"] + [man[k] for k in ("vision", "audio") if k in man]:
        f = st["file"]
        place = comp.get("placement", {}).get("stages", {}).get(f, {})
        what = ("vision tower + projector" if f.startswith("vision") else "audio encoder + adapter" if f.startswith("audio")
                else f"trunk layers {st['layers'][0]}–{st['layers'][-1]} + head")
        rows.append([f"`{f}`", what, len(st["entries"]), f"{st['mb']:.0f} MB", f"{st['convert_s']:.0f} s",
                     f"{comp.get('seconds', {}).get(f, float('nan')):.0f} s", place.get("status", "?"),
                     ",".join(map(str, place.get("modes", [])))])
    return (table(["Package", "Contents", "Entries", "Size", "Convert", "Compile (cold)", "Placement", "Bonded mode"], rows,
                  "llrrrrlr")
            + f"\nSequence widths: {', '.join(map(str, man['widths']))}. "
            f"Vision crops: {', '.join(map(str, man.get('vision', {}).get('widths', [])))} patches. "
            f"Audio: {', '.join(map(str, man.get('audio', {}).get('widths', [])))} positions (80 ms each). "
            f"{man['quant'].upper() if man['quant'] == 'fp16' else man['quant']} weights, SiLU as `{man['silu']}`. "
            "The embedding table (128 MB, FP16) stays on the host.\n")


def latency_section(res: dict) -> str:
    by = {e: {r["tokens"]: r for r in d["rows"]} for e, d in res.items()}
    lengths = sorted({t for d in by.values() for t in d})
    engines = [e for e in ORDER if e in by]
    rows = []
    for L in lengths:
        a = by.get("coreai-ane", {}).get(L)
        row = [L, a["width"] if a else "–"] + [ms(by[e][L]["median_ms"]) if L in by[e] else "–" for e in engines]
        for other in VERSUS:
            if other in by:
                row.append(f"{by[other][L]['median_ms'] / a['median_ms']:.1f}×" if a and L in by[other] else "–")
        rows.append(row)
    head = ["Tokens", "ANE width"] + [NAMES[e] for e in engines] + [f"{NAMES[o]} / ANE" for o in VERSUS if o in by]
    out = table(head, rows, "r" * len(head))
    if "coreai-ane" in by:
        full = [(L, r) for L, r in by["coreai-ane"].items() if r["width"] == L]
        out += "\nANE cost per compiled width (a decision pays for the whole width it runs in):\n\n" + table(
            ["Width", "Median", "p95", "Decisions/s", "µs per token slot"],
            [[L, f"{r['median_ms']:.2f} ms", f"{r['p95_ms']:.2f} ms", f"{1e3 / r['median_ms']:.0f}",
              f"{1e3 * r['median_ms'] / L:.0f}"] for L, r in sorted(full)], "rrrrr")
    return out


def dataset_section(res: dict) -> str:
    if REF not in res:
        return "_no FP32 reference run_\n"
    ref = {r["id"]: r for r in res[REF]["rows"] if "error" not in r}
    rows = [[NAMES[e]] + parity(res[e]["rows"], ref, e == REF, len(ref) / res[e]["wall_s"]) for e in ORDER if e in res]
    tok = np.array([r["tokens"] for r in ref.values()])
    return table(["Engine"] + PARITY_HEAD, rows) + (
        f"\nPrompt lengths: median {int(np.median(tok))}, p90 {int(np.percentile(tok, 90))}, max {tok.max()} tokens; "
        f"{len({r['domain'] for r in ref.values()})} domains. Decisions/s is the whole loop on one thread "
        "(tokenizer, embedding lookup, forward, softmax), one decision at a time, in a single pass. PyTorch MPS "
        "specializes its graph for every new sequence length, so a first pass over rows of varying length is slower than "
        "the fixed-length sweep above: a repeated pass over 400 of these rows ran at a 36 ms median with MPS fp16 "
        "(39 ms on the first). The Core AI entries are fixed shapes compiled ahead of time.\n")


TETRIS = [   # (file tag, who, sentence, question)
    ("d1-int8_calibrated-rating", "d1-omni-600M, int8", "calibrated to d1", "five-level rating"),
    ("laya_graded", "laya", "graded (tuned to laya)", "clean placement?"),
    ("laya_plain", "laya", "plain (the original demo)", "clean placement?"),
    ("d1-int8_graded", "d1-omni-600M, int8", "graded (tuned to laya)", "clean placement?"),
    ("d1-int8_plain", "d1-omni-600M, int8", "plain (the original demo)", "clean placement?"),
    ("laya_calibrated-rating", "laya", "calibrated to d1", "five-level rating"),
    ("laya_calibrated-clean", "laya", "calibrated to d1", "clean placement?"),
]


def tetris_section(results: Path) -> str:
    out = ""
    for suffix, title in (("-shortlist", "With the harness filter (a landing that buries a cell is withheld when a clean one exists)"),
                          ("-shortlist_seeds11-20", "With the filter, on seeds 11 to 20 (d1's wording and question were "
                                                    "chosen with games of seeds 1 to 10 in view; these seeds were not)"),
                          ("", "Without the filter")):
        rows = []
        for tag, who, sentence, question in TETRIS:
            f = results / f"tetris_{tag}{suffix}.json"
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            s, games = d["summary"], d["games"]
            rows.append([who, sentence, question, f"{s['mean_pieces']:,.1f}", f"{s['mean_lines']:,.1f}",
                         f"{s['at_cap']}/{len(games)}", f"{100 * s['heuristic_agreement']:.0f}%",
                         f"{s['landing_ms']:.2f} ms", f"{s['decisions_per_min']:,}"])
        if rows:
            out += f"{title}:\n\n" + table(["Model", "Sentence", "Question", "Mean pieces", "Mean lines", "Reached the cap",
                                            "Same pick as the heuristic", "Per decision", "Decisions/min"], rows, "lllrrrrrr") + "\n"
    return out


def quant_section(q: dict) -> str:
    cell = lambda r, n: f"{r[n + '_ms']:.1f} ms ({r.get(n + '_p10_ms', r[n + '_ms']):.1f})"  # noqa: E731
    return table(["Width", "FP16 weights", "int8 weights", "FP16 / int8 (10th percentiles)"],
                 [[r["width"], cell(r, "fp16"), cell(r, "int8"),
                   f"{r.get('fp16_p10_ms', r['fp16_ms']) / r.get('int8_p10_ms', r['int8_ms']):.2f}×"] for r in q["rows"]], "rrrr")


def speed_section(sp: dict) -> str:
    names = {"d1 int8": "d1-omni-600M, int8 weights", "d1 fp16": "d1-omni-600M, FP16 weights",
             "laya": "laya-multilingual, FP16 (their Core ML package)"}
    return table(["Model", "Decisions", "Median", "Mean", "p95", "Decisions/min"],
                 [[names.get(r["player"], r["player"]), f"{r['decisions']:,}", f"{r['median_ms']:.2f} ms",
                   f"{r['mean_ms']:.2f} ms", f"{r['p95_ms']:.2f} ms", f"{r['decisions_per_min']:,}"] for r in sp["rows"]])


def http_section(res: dict) -> str:
    return table(["Request", "Questions", "Tokens", "HTTP median", "HTTP p95", "Inside the server", "ANE", "Requests/s"],
                 [[r["request"], r["questions"], r["tokens"], f"{r['http_median_ms']:.2f} ms", f"{r['http_p95_ms']:.2f} ms",
                   f"{r['server_ms']:.2f} ms", f"{r['ane_ms']:.2f} ms", f"{r['requests_per_s']:.0f}"] for r in res["rows"]])


def sweep_section(res: dict, key: str, label: str, first: list[tuple[str, str]], pre: str) -> str:
    """End-to-end medians per engine at each image size / clip length, with the ANE's stage breakdown."""
    engines = [e for e in ORDER if e in res]
    by = {e: {r[key]: r for r in res[e]["rows"]} for e in engines}
    rows = []
    for k in [r[key] for r in res[engines[0]]["rows"]]:
        a = by.get("coreai-ane", {}).get(k, {})
        rows.append([f"{k} s" if key == "seconds" else k] + [a.get(c, "–") for c, _ in first]
                    + [ms(by[e][k]["median_ms"]) for e in engines]
                    + [ms(a.get("preprocess_ms")), ms(a.get("encoder_ms")), ms(a.get("trunk_ms"))])
    head = [label] + [h for _, h in first] + [NAMES[e] for e in engines] + [f"ANE: {pre}", "ANE: encoder", "ANE: trunk + head"]
    return table(head, rows)


def tower_section(res: dict) -> str:
    engines = [e for e in ORDER if res.get(e, {}).get("tower")]
    by = {e: {r["patches"]: r for r in res[e]["tower"]} for e in engines}
    return table(["Crop (patches)", "Prefix positions"] + [NAMES[e] for e in engines],
                 [[n, n // 4] + [f"{by[e][n]['median_ms']:.1f} ms" for e in engines] for n in sorted(by[engines[0]])])


def sets_section(res: dict) -> str:
    """Parity and accuracy per data set and engine."""
    if REF not in res:
        return "_no FP32 reference run_\n"
    ref = {r["id"]: r for r in res[REF]["rows"]}
    sets = list(dict.fromkeys(r["set"] for r in res[REF]["rows"]))
    rows = []
    for s in sets + (["all"] if len(sets) > 1 else []):
        for e in [x for x in ORDER if x in res]:
            mine = [r for r in res[e]["rows"] if s == "all" or r["set"] == s]
            rows.append([s, NAMES[e]] + parity(mine, ref, e == REF))
    return table(["Set", "Engine"] + PARITY_HEAD, rows, "ll" + "r" * len(PARITY_HEAD))


def stage_means(res: dict) -> str:
    rows = res["coreai-ane"]["rows"]
    mean = lambda k: np.mean([r[k] for r in rows])  # noqa: E731
    return (f"host preprocessing {mean('preprocess_ms'):.1f} ms, encoder {mean('encoder_ms'):.1f} ms, "
            f"trunk + head {mean('trunk_ms'):.1f} ms, {mean('tokens'):.0f} positions")


DESIGN = """## Design checks

Measured on the ANE on earlier builds of the same day; the result files were not kept.

Trunk and head, two-width builds (128 and 512):

| Variant | 128 tokens | 512 tokens | Argmax ≠ FP32 (2,600 text decisions) | Kept |
| --- | ---: | ---: | ---: | --- |
| One package, SiLU as `x/2 · (1 + tanh(x/2))` | 7.9 ms | 46.8 ms | 6 | yes |
| Four 4-layer packages (system1-ane's layout), same SiLU | 8.8 ms | 48.2 ms | not run | no: one host round trip per package |
| One package, the ANE's native SiLU | 7.0 ms | 44.2 ms | 51 | no: 8% faster, 8× the changed answers |

GELU in the media encoders (the first media benchmark pass used the ANE's own GELU; the tables above use the kept form):

| Variant | Encoder time | Argmax ≠ FP32 | mean KL | Kept |
| --- | ---: | ---: | ---: | --- |
| Audio encoder, the ANE's native GELU | 42.7 ms (30 s clip) | 17 of 763 spoken intents | 4.5e-03 | no |
| Audio encoder, GELU written out with erf | 43.7 ms | 8 of 763 | 4.7e-04 | yes: prefix error against FP32 4.3% → 0.4% |
| Vision tower, the ANE's native tanh-GELU | 61.8 ms (1,024-patch crop) | 0 of 600 POPE questions | 2.3e-04 | no |
| Vision tower, tanh-GELU written out | 64.5 ms | 0 of 600 | 1.7e-04 | yes: 4% slower, worst prefix cosine on a tiled image 0.985 → 0.995 |

`--compute auto` follows the tables above: trunk and head on the ANE up to 512 positions and on the GPU beyond; the
vision tower on the ANE up to 512 patches per crop and on the GPU for the 768- and 1,024-patch crops; the audio encoder
on the GPU for its 30 s entry and on the ANE below it. One check of it against the ANE alone (medians of 10 runs, end
to end): a 640x480 photo 70.6 ms against 91.6 ms, the tiled 1600x1200 image 493 ms against 797 ms, a 30 s clip 83.4 ms
against 93.6 ms, a 320x240 image unchanged (29.8 ms).

Making short decisions faster (13 Oct, trunk and head at 64 positions, one build each, timed one after the other):

| Variant | 64 positions | Package | Kept |
| --- | ---: | ---: | --- |
| FP16 weights | 5.9 ms | 628 MB | yes: the reference build |
| int8 weights, linear per output channel | 3.35 to 3.45 ms | 316 MB | yes: `--quant int8` |
| int4 weights, linear per output channel | 3.3 ms | 158 MB | no: no faster than int8, accuracy not measured |
| 4-bit palette per tensor | 3.4 to 3.7 ms | 157 MB | no: no faster than int8, accuracy not measured |
| int8, four sequences in one call (a batch axis) | 16.4 ms, 4.1 ms per sequence | 316 MB | no: the ANE runs a batch one sequence after the other; 8 and 16 scale the same |

What a call costs on this chip, from these and the width tables: FP16 is flat at 5.9 ms from 16 to 64 positions, so
below about 100 positions the time goes into reading the weights (628 MB per call); int8 halves that. From there on the
cost is the positions, about 56 µs each, which is why 4-bit weights add nothing at 64 positions and why int8 is no
faster than FP16 from 128 positions up. Widths that are not a multiple of 32 were slower than the next one up in the
same build (48 and 56 against 64, 72 and 80 against 96), so none are compiled.

Activations were left in FP16. Rounding the convolution inputs to 8 bits as well was tried: it engages a faster path
from about 192 positions (11 ms against 16.5 ms at 256) and none below, and it changed 4% of the benchmark answers
even after a per-channel equalization and calibration on the benchmark's own domains. It is not in this tree.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=ROOT / "bench/results")
    ap.add_argument("--build", type=Path, default=ROOT / "build/d1-omni-600M-fp16", help="the FP16 reference build")
    a = ap.parse_args(argv)
    res = {t: load(a.results, t) for t in ("latency", "dataset", "vision", "pope", "audio", "speech")}
    m = next(iter(res["latency"].values()))["machine"]
    print("# Measured results: d1-omni-600M on the Apple Neural Engine\n")
    print(f"{m['chip']} ({m['model']}), macOS {m['macos']} ({m['build']}), Python {m['python']}, coreai-core 1.0.0b2. "
          f"Measured {m['time'][:10]} on a machine that was not idle; one engine at a time, medians of warm runs. "
          "Not a controlled benchmark. The comparisons are the same packages on the GPU (Core AI) and the checkpoint in "
          "PyTorch on the GPU (MPS); PyTorch MPS FP32 is the numerical reference. The CPU is not benchmarked.\n")
    print("## Build\n\nThe FP16 reference build (`d1ane setup --quant fp16`). The default build (`d1ane setup`) is the same "
          "packages with the trunk and head weights in int8.\n\n" + build_section(a.build))
    if res["latency"]:
        print("## Text: latency of one decision (ms, median)\n\nOne question over one state, embedding lookup to option "
              "logits, tokenizer excluded. The ANE runs the narrowest compiled width that holds the prompt; PyTorch runs "
              "the exact length.\n\n" + latency_section(res["latency"]))
    if res["dataset"]:
        print("## Text: parity and accuracy on fast-decisions\n\nEvery single-label head of "
              "[fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions) (development split), each "
              "as a d1 `choice` question whose options are the label names.\n\n" + dataset_section(res["dataset"]))
    if (a.results / "quant_ane.json").is_file():
        q = json.loads((a.results / "quant_ane.json").read_text())
        print(f"## Text: int8 weights (`--quant int8`)\n\nThe trunk and head weights stored in 8 bits, linear per output "
              f"channel, activations in FP16 as before: {q['package_mb']['int8']} MB against {q['package_mb']['fp16']} MB, "
              "every entry still fully on the ANE. Its accuracy is the \"int8 weights\" row of the tables. One forward "
              f"per width, both builds in one process and timed in alternating rounds ({q['rounds']} rounds of {q['calls']} "
              f"calls, measured {q['time'][:10]}, load average {q['load_avg'][0]}); median, and the 10th percentile in "
              "brackets, which moves less with the load:\n\n" + quant_section(q))
    if (a.results / "http_coreai-ane.json").is_file():
        http = json.loads((a.results / "http_coreai-ane.json").read_text())
        print(f"## Text: server (`POST /v1/systemone`, ANE)\n\n{http['n']} requests each over one keep-alive connection, "
              "after 10 warm-up requests. A request with several questions runs one forward per question.\n\n"
              + http_section(http))
    if res["vision"]:
        print("## Images: one question, end to end (ms, median)\n\nA COCO photo resized to four sizes, one 3-option "
              "question: tiling and patchifying on the host, the vision encoder (one tower call and one projector call "
              "per crop), then trunk and head over the image prefix and the question.\n\n"
              + sweep_section(res["vision"], "image", "Image", [("crops", "Crops"), ("tokens", "Positions")], "preprocess"))
        print("The vision encoder alone, per crop (tower + 2×2 unshuffle on the host + projector):\n\n"
              + tower_section(res["vision"]))
    if res["pope"]:
        print("## Images: parity and accuracy on POPE\n\n600 yes/no object questions of "
              "[lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE) on COCO photos (200 spread evenly over each "
              "category), each a d1 `noul` question with the photo as the whole state. Timed from the decoded image to "
              "the probabilities.\n\n" + sets_section(res["pope"]))
        if "coreai-ane" in res["pope"]:
            print(f"Mean ANE time per question: {stage_means(res['pope'])}.\n")
    if res["audio"]:
        print("## Audio: one question, end to end (ms, median)\n\nA speech clip looped and cut to five lengths, one "
              "3-option question: log-mel features on the host (FP32), the audio encoder (one call), then trunk and "
              "head over the audio prefix and the question.\n\n"
              + sweep_section(res["audio"], "seconds", "Clip", [("encoder_width", "Encoder width"), ("tokens", "Positions")],
                              "log-mel"))
    if res["speech"]:
        print("## Audio: parity and accuracy on spoken intents\n\nThe 200 SLURP clips of "
              "[DynamicSuperb/SuperbIC_SLURP-Intent](https://huggingface.co/datasets/DynamicSuperb/SuperbIC_SLURP-Intent) "
              "(5 intents) and the 563 en-US clips of [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) "
              "(14 banking intents, 8 kHz telephone audio resampled to 16 kHz, cut at 30 s by the model). Each is a d1 "
              "`choice` question whose options are the intent names, with the clip as the whole state. Timed from the "
              "decoded samples to the probabilities, in a single pass: PyTorch MPS specializes for every new clip length "
              "(hence its p95), so its median is the fairer per-decision figure. MINDS-14 is far from the assistant "
              "requests the model's audio was trained on, and every engine scores about 21% on it; it is here to compare "
              "engines.\n\n" + sets_section(res["speech"]))
        if "coreai-ane" in res["speech"]:
            print(f"Mean ANE time per question: {stage_means(res['speech'])}.\n")
    if any(a.results.glob("tetris_*.json")):
        print("## Tetris: decisions as a game\n\nThe harness of FluidUse's laya Tetris demo, ported (`d1ane/tetris.py`; its "
              "heuristic control reproduces their published 582.8 and 2,874.4 mean pieces). Every legal landing is one "
              "sentence, the model rates it in one call, the best-rated landing is played. Seeds 1 to 10, a 5,000-piece cap, "
              "one call per landing, nothing cached. laya is their published Core ML package (L128, CPU + ANE) called from "
              "Python, not their Swift runtime. \"Per decision\" is the mean time of a call inside the game, tokenizer "
              "included; these games ran one after the other on a busy machine, so compare speeds in the next table."
              "\n\n" + tetris_section(a.results))
        if (a.results / "tetris_speed.json").is_file():
            sp = json.loads((a.results / "tetris_speed.json").read_text())
            print(f"Speed, measured in turns: each model plays {sp['pieces']} pieces of a seed alone with the filter on, "
                  f"then the next model plays the same seed, for {sp['rounds']} seeds (load average {sp['load_avg'][0]}). "
                  "A decision is one landing rated: tokenizer, host arrays, the model call and the readout.\n\n"
                  + speed_section(sp))
    print(DESIGN)


if __name__ == "__main__":
    main()
