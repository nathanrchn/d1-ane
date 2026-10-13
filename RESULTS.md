# Measured results: d1-omni-600M on the Apple Neural Engine

Apple M4 Max (Mac16,6), macOS 27.2 (26B5091g), Python 3.12.12, coreai-core 1.0.0b2. Measured 2026-10-10 (sections added later give their own date) on a machine that was not idle; one engine at a time, medians of warm runs. Not a controlled benchmark. The comparisons are the same packages on the GPU (Core AI) and the checkpoint in PyTorch on the GPU (MPS); PyTorch MPS FP32 is the numerical reference. The CPU is not benchmarked.

## Build

The FP16 reference build (`d1ane setup --quant fp16`). The default build (`d1ane setup`) is the same packages with the trunk and head weights in int8.

| Package | Contents | Entries | Size | Convert | Compile (cold) | Placement | Bonded mode |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| `stage0_L00-15_head.aimodel` | trunk layers 0–15 + head | 10 | 640 MB | 23 s | 157 s | fully_ane | 0 |
| `vision.aimodel` | vision tower + projector | 8 | 188 MB | 9 s | 23 s | fully_ane | 0 |
| `audio.aimodel` | audio encoder + adapter | 4 | 245 MB | 19 s | 36 s | fully_ane | 0 |

Sequence widths: 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048. Vision crops: 256, 512, 768, 1024 patches. Audio: 64, 128, 256, 376 positions (80 ms each). FP16 weights, SiLU as `tanh`. The embedding table (128 MB, FP16) stays on the host.

## Text: latency of one decision (ms, median)

One question over one state, embedding lookup to option logits, tokenizer excluded. The ANE runs the narrowest compiled width that holds the prompt; PyTorch runs the exact length.

| Tokens | ANE width | Core AI, ANE | Core AI, GPU | PyTorch MPS fp16 | Core AI, GPU / ANE | PyTorch MPS fp16 / ANE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 64 | 5.9 | 11.0 | 16.7 | 1.9× | 2.8× |
| 64 | 64 | 5.9 | 10.9 | 16.0 | 1.8× | 2.7× |
| 96 | 128 | 7.7 | 15.3 | 15.1 | 2.0× | 2.0× |
| 128 | 128 | 7.7 | 15.5 | 18.3 | 2.0× | 2.4× |
| 192 | 192 | 13.5 | 20.0 | 24.7 | 1.5× | 1.8× |
| 256 | 256 | 15.5 | 25.4 | 30.0 | 1.6× | 1.9× |
| 384 | 384 | 25.2 | 35.4 | 42.8 | 1.4× | 1.7× |
| 512 | 512 | 46.4 | 45.4 | 55.1 | 1.0× | 1.2× |
| 768 | 768 | 79.5 | 66.2 | 83.8 | 0.8× | 1.1× |
| 1024 | 1024 | 116 | 87.5 | 115 | 0.8× | 1.0× |
| 1536 | 1536 | 206 | 132 | 183 | 0.6× | 0.9× |
| 2048 | 2048 | 333 | 193 | 262 | 0.6× | 0.8× |

ANE cost per compiled width (a decision pays for the whole width it runs in):

| Width | Median | p95 | Decisions/s | µs per token slot |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 5.91 ms | 5.97 ms | 169 | 92 |
| 128 | 7.70 ms | 7.77 ms | 130 | 60 |
| 192 | 13.46 ms | 13.51 ms | 74 | 70 |
| 256 | 15.52 ms | 15.68 ms | 64 | 61 |
| 384 | 25.23 ms | 25.29 ms | 40 | 66 |
| 512 | 46.43 ms | 46.55 ms | 22 | 91 |
| 768 | 79.53 ms | 79.68 ms | 13 | 104 |
| 1024 | 115.53 ms | 115.68 ms | 9 | 113 |
| 1536 | 205.97 ms | 206.30 ms | 5 | 134 |
| 2048 | 332.78 ms | 333.20 ms | 3 | 162 |

## Text: parity and accuracy on fast-decisions

Every single-label head of [fastino/fast-decisions](https://huggingface.co/datasets/fastino/fast-decisions) (development split), each as a d1 `choice` question whose options are the label names.

| Engine | Decisions | Accuracy | Argmax ≠ FP32 | mean max\|Δp\| | p99 max\|Δp\| | mean KL | Median ms | p95 ms | Decisions/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Core AI, ANE | 2600 | 75.31% | 6 (0.23%) | 2.2e-03 | 1.3e-02 | 4.7e-05 | 15.8 | 26.4 | 58 |
| Core AI, ANE, int8 weights | 2600 | 75.23% | 23 (0.88%) | 8.9e-03 | 5.3e-02 | 7.0e-04 | 17.2 | 27.9 | 55 |
| Core AI, GPU | 2600 | 75.19% | 2 (0.08%) | 8.9e-04 | 5.8e-03 | 7.2e-06 | 27.0 | 41.1 | 36 |
| PyTorch MPS fp16 | 2600 | 75.12% | 6 (0.23%) | 6.8e-04 | 4.0e-03 | 4.1e-06 | 45.0 | 118.7 | 16 |
| PyTorch MPS fp32 | 2600 | 75.19% | reference | – | – | – | 37.9 | 82.0 | 23 |

Prompt lengths: median 206, p90 275, max 444 tokens; 17 domains. Decisions/s is the whole loop on one thread (tokenizer, embedding lookup, forward, softmax), one decision at a time, in a single pass. PyTorch MPS specializes its graph for every new sequence length, so a first pass over rows of varying length is slower than the fixed-length sweep above: a repeated pass over 400 of these rows ran at a 36 ms median with MPS fp16 (39 ms on the first). The Core AI entries are fixed shapes compiled ahead of time.

## Text: int8 weights (`--quant int8`)

The trunk and head weights stored in 8 bits, linear per output channel, activations in FP16 as before: 327 MB against 640 MB, every entry still fully on the ANE. Its accuracy is the "int8 weights" row of the tables. One forward per width, both builds in one process and timed in alternating rounds (8 rounds of 40 calls, measured 2026-10-13, load average 4.5); median, and the 10th percentile in brackets, which moves less with the load:

| Width | FP16 weights | int8 weights | FP16 / int8 (10th percentiles) |
| ---: | ---: | ---: | ---: |
| 64 | 5.9 ms (5.8) | 3.4 ms (3.3) | 1.74× |
| 128 | 7.7 ms (7.6) | 7.3 ms (7.2) | 1.05× |
| 192 | 13.5 ms (13.4) | 13.3 ms (13.1) | 1.02× |
| 256 | 15.6 ms (15.4) | 16.7 ms (16.5) | 0.93× |
| 384 | 25.2 ms (25.0) | 26.2 ms (26.0) | 0.96× |
| 512 | 46.4 ms (46.2) | 37.9 ms (37.7) | 1.23× |

## Text: server (`POST /v1/systemone`, ANE)

200 requests each over one keep-alive connection, after 10 warm-up requests. A request with several questions runs one forward per question.

| Request | Questions | Tokens | HTTP median | HTTP p95 | Inside the server | ANE | Requests/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| route a ticket (1 choice, 4 options) | 1 | 63 | 6.11 ms | 6.64 ms | 5.96 ms | 5.86 ms | 162 |
| guardrail (1 noul) | 1 | 47 | 6.07 ms | 6.17 ms | 5.93 ms | 5.84 ms | 165 |
| three questions, one request | 3 | 154 | 18.29 ms | 18.73 ms | 18.05 ms | 17.70 ms | 55 |

## Images: one question, end to end (ms, median)

A COCO photo resized to four sizes, one 3-option question: tiling and patchifying on the host, the vision encoder (one tower call and one projector call per crop), then trunk and head over the image prefix and the question.

| Image | Crops | Positions | Core AI, ANE | Core AI, GPU | PyTorch MPS fp16 | ANE: preprocess | ANE: encoder | ANE: trunk + head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 224x224 | 1 | 94 | 17.2 | 28.7 | 55.4 | 0.9 | 8.5 | 7.6 |
| 320x240 | 1 | 110 | 30.5 | 40.1 | 57.6 | 0.8 | 21.7 | 7.7 |
| 640x480 | 1 | 264 | 92.2 | 80.2 | 72.8 | 1.5 | 64.6 | 25.3 |
| 1600x1200 | 7 | 1800 | 797 | 496 | 463 | 10.1 | 453 | 331 |

The vision encoder alone, per crop (tower + 2×2 unshuffle on the host + projector):

| Crop (patches) | Prefix positions | Core AI, ANE | Core AI, GPU |
| --- | ---: | ---: | ---: |
| 256 | 64 | 8.0 ms | 12.2 ms |
| 512 | 128 | 21.6 ms | 22.5 ms |
| 768 | 192 | 40.4 ms | 33.3 ms |
| 1024 | 256 | 64.5 ms | 45.0 ms |

## Images: parity and accuracy on POPE

600 yes/no object questions of [lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE) on COCO photos (200 spread evenly over each category), each a d1 `noul` question with the photo as the whole state. Timed from the decoded image to the probabilities.

| Set | Engine | Decisions | Accuracy | Argmax ≠ FP32 | mean max\|Δp\| | p99 max\|Δp\| | mean KL | Median ms | p95 ms | Decisions/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| POPE random | Core AI, ANE | 200 | 87.00% | 0 (0.00%) | 1.5e-03 | 4.7e-02 | 2.2e-04 | 92.4 | 94.3 | 12 |
| POPE random | Core AI, ANE, int8 weights | 200 | 87.00% | 0 (0.00%) | 3.0e-03 | 7.8e-02 | 8.3e-04 | 94.0 | 98.0 | 11 |
| POPE random | PyTorch MPS fp16 | 200 | 87.00% | 0 (0.00%) | 4.7e-04 | 1.1e-02 | 3.1e-05 | 78.6 | 108.6 | 12 |
| POPE random | PyTorch MPS fp32 | 200 | 87.00% | reference | – | – | – | 85.3 | 110.9 | 11 |
| POPE popular | Core AI, ANE | 200 | 85.00% | 0 (0.00%) | 1.5e-03 | 4.7e-02 | 1.4e-04 | 92.4 | 104.4 | 11 |
| POPE popular | Core AI, ANE, int8 weights | 200 | 84.50% | 1 (0.50%) | 4.1e-03 | 1.0e-01 | 1.0e-03 | 93.9 | 97.7 | 11 |
| POPE popular | PyTorch MPS fp16 | 200 | 85.00% | 0 (0.00%) | 5.3e-04 | 1.2e-02 | 2.1e-05 | 77.5 | 88.8 | 13 |
| POPE popular | PyTorch MPS fp32 | 200 | 85.00% | reference | – | – | – | 90.7 | 94.5 | 11 |
| POPE adversarial | Core AI, ANE | 200 | 79.50% | 0 (0.00%) | 1.6e-03 | 2.7e-02 | 1.6e-04 | 92.5 | 94.0 | 12 |
| POPE adversarial | Core AI, ANE, int8 weights | 200 | 79.00% | 1 (0.50%) | 4.3e-03 | 1.0e-01 | 1.0e-03 | 93.5 | 94.7 | 11 |
| POPE adversarial | PyTorch MPS fp16 | 200 | 79.50% | 0 (0.00%) | 5.0e-04 | 1.1e-02 | 1.7e-05 | 77.4 | 88.0 | 13 |
| POPE adversarial | PyTorch MPS fp32 | 200 | 79.50% | reference | – | – | – | 87.8 | 92.7 | 11 |
| all | Core AI, ANE | 600 | 83.83% | 0 (0.00%) | 1.5e-03 | 4.7e-02 | 1.7e-04 | 92.4 | 94.5 | 11 |
| all | Core AI, ANE, int8 weights | 600 | 83.50% | 2 (0.33%) | 3.8e-03 | 1.0e-01 | 9.5e-04 | 93.7 | 96.7 | 11 |
| all | PyTorch MPS fp16 | 600 | 83.83% | 0 (0.00%) | 5.0e-04 | 1.2e-02 | 2.3e-05 | 77.9 | 96.0 | 13 |
| all | PyTorch MPS fp32 | 600 | 83.83% | reference | – | – | – | 89.1 | 94.9 | 11 |

Mean ANE time per question: host preprocessing 1.7 ms, encoder 61.6 ms, trunk + head 23.6 ms, 254 positions.

## Audio: one question, end to end (ms, median)

A speech clip looped and cut to five lengths, one 3-option question: log-mel features on the host (FP32), the audio encoder (one call), then trunk and head over the audio prefix and the question.

| Clip | Encoder width | Positions | Core AI, ANE | Core AI, GPU | PyTorch MPS fp16 | ANE: log-mel | ANE: encoder | ANE: trunk + head |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 s | 64 | 72 | 11.6 | 26.8 | 40.6 | 0.6 | 3.2 | 7.7 |
| 5 s | 64 | 110 | 12.4 | 27.7 | 39.7 | 1.4 | 3.1 | 7.7 |
| 10 s | 128 | 172 | 20.5 | 33.7 | 47.4 | 1.7 | 5.2 | 13.4 |
| 20 s | 256 | 297 | 45.3 | 54.4 | 64.2 | 2.3 | 17.5 | 25.1 |
| 30 s | 376 | 422 | 93.8 | 70.3 | 80.2 | 3.3 | 43.7 | 46.1 |

## Audio: parity and accuracy on spoken intents

The 200 SLURP clips of [DynamicSuperb/SuperbIC_SLURP-Intent](https://huggingface.co/datasets/DynamicSuperb/SuperbIC_SLURP-Intent) (5 intents) and the 563 en-US clips of [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) (14 banking intents, 8 kHz telephone audio resampled to 16 kHz, cut at 30 s by the model). Each is a d1 `choice` question whose options are the intent names, with the clip as the whole state. Timed from the decoded samples to the probabilities, in a single pass: PyTorch MPS specializes for every new clip length (hence its p95), so its median is the fairer per-decision figure. MINDS-14 is far from the assistant requests the model's audio was trained on, and every engine scores about 21% on it; it is here to compare engines.

| Set | Engine | Decisions | Accuracy | Argmax ≠ FP32 | mean max\|Δp\| | p99 max\|Δp\| | mean KL | Median ms | p95 ms | Decisions/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SLURP intent | Core AI, ANE | 200 | 82.00% | 2 (1.00%) | 1.6e-03 | 1.7e-02 | 8.9e-05 | 12.2 | 19.9 | 78 |
| SLURP intent | Core AI, ANE, int8 weights | 200 | 82.50% | 1 (0.50%) | 5.4e-03 | 4.6e-02 | 7.0e-04 | 12.8 | 20.7 | 74 |
| SLURP intent | PyTorch MPS fp16 | 200 | 83.00% | 0 (0.00%) | 7.9e-04 | 7.1e-03 | 6.5e-05 | 62.0 | 200.3 | 10 |
| SLURP intent | PyTorch MPS fp32 | 200 | 83.00% | reference | – | – | – | 53.4 | 194.9 | 10 |
| MINDS-14 en-US | Core AI, ANE | 563 | 20.78% | 6 (1.07%) | 8.1e-03 | 5.5e-02 | 6.0e-04 | 22.5 | 90.0 | 33 |
| MINDS-14 en-US | Core AI, ANE, int8 weights | 563 | 20.60% | 14 (2.49%) | 2.8e-02 | 1.6e-01 | 6.4e-03 | 24.5 | 82.6 | 32 |
| MINDS-14 en-US | PyTorch MPS fp16 | 563 | 20.78% | 3 (0.53%) | 2.8e-03 | 1.7e-02 | 7.6e-05 | 77.9 | 519.0 | 6 |
| MINDS-14 en-US | PyTorch MPS fp32 | 563 | 20.60% | reference | – | – | – | 76.1 | 526.6 | 6 |
| all | Core AI, ANE | 763 | 36.83% | 8 (1.05%) | 6.4e-03 | 4.9e-02 | 4.7e-04 | 21.7 | 46.9 | 39 |
| all | Core AI, ANE, int8 weights | 763 | 36.83% | 15 (1.97%) | 2.2e-02 | 1.5e-01 | 4.9e-03 | 22.9 | 48.4 | 38 |
| all | PyTorch MPS fp16 | 763 | 37.09% | 3 (0.39%) | 2.3e-03 | 1.6e-02 | 7.4e-05 | 76.8 | 508.9 | 7 |
| all | PyTorch MPS fp32 | 763 | 36.96% | reference | – | – | – | 73.6 | 521.3 | 7 |

Mean ANE time per question: host preprocessing 1.5 ms, encoder 7.7 ms, trunk + head 16.1 ms, 204 positions.

## Tetris: decisions as a game

The harness of FluidUse's laya Tetris demo, ported (`d1ane/tetris.py`; its heuristic control reproduces their published 582.8 and 2,874.4 mean pieces). Every legal landing is one sentence, the model rates it in one call, the best-rated landing is played. Seeds 1 to 10, a 5,000-piece cap, one call per landing, nothing cached. laya is their published Core ML package (L128, CPU + ANE) called from Python, not their Swift runtime. "Per decision" is the mean time of a call inside the game, tokenizer included; these games ran one after the other on a busy machine, so compare speeds in the next table.

With the harness filter (a landing that buries a cell is withheld when a clean one exists):

| Model | Sentence | Question | Mean pieces | Mean lines | Reached the cap | Same pick as the heuristic | Per decision | Decisions/min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1-omni-600M, int8 | calibrated to d1 | five-level rating | 3,048.1 | 1,210.1 | 3/10 | 87% | 3.73 ms | 16,068 |
| laya | graded (tuned to laya) | clean placement? | 568.3 | 214.5 | 0/10 | 68% | 4.13 ms | 14,508 |
| laya | plain (the original demo) | clean placement? | 78.6 | 18.8 | 0/10 | 55% | 4.12 ms | 14,493 |
| d1-omni-600M, int8 | graded (tuned to laya) | clean placement? | 49.9 | 7.7 | 0/10 | 44% | 5.41 ms | 11,358 |
| d1-omni-600M, int8 | plain (the original demo) | clean placement? | 56.1 | 10.3 | 0/10 | 52% | 4.13 ms | 14,064 |
| laya | calibrated to d1 | five-level rating | 22.6 | 0.1 | 0/10 | 20% | 4.37 ms | 13,790 |
| laya | calibrated to d1 | clean placement? | 154.4 | 49.5 | 0/10 | 58% | 4.40 ms | 13,591 |

With the filter, on seeds 11 to 20 (d1's wording and question were chosen with games of seeds 1 to 10 in view; these seeds were not):

| Model | Sentence | Question | Mean pieces | Mean lines | Reached the cap | Same pick as the heuristic | Per decision | Decisions/min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1-omni-600M, int8 | calibrated to d1 | five-level rating | 2,420.2 | 958.2 | 3/10 | 87% | 4.35 ms | 14,201 |
| laya | graded (tuned to laya) | clean placement? | 492.4 | 184.2 | 0/10 | 67% | 4.52 ms | 13,200 |

Without the filter:

| Model | Sentence | Question | Mean pieces | Mean lines | Reached the cap | Same pick as the heuristic | Per decision | Decisions/min |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d1-omni-600M, int8 | calibrated to d1 | five-level rating | 599.5 | 224.7 | 0/10 | 78% | 4.19 ms | 14,050 |
| laya | graded (tuned to laya) | clean placement? | 56.0 | 8.5 | 0/10 | 48% | 4.05 ms | 14,844 |
| laya | plain (the original demo) | clean placement? | 91.7 | 23.3 | 0/10 | 45% | 4.29 ms | 14,000 |
| d1-omni-600M, int8 | graded (tuned to laya) | clean placement? | 43.0 | 4.3 | 0/10 | 42% | 4.99 ms | 12,013 |
| d1-omni-600M, int8 | plain (the original demo) | clean placement? | 60.0 | 11.0 | 0/10 | 37% | 4.21 ms | 14,638 |


Speed, measured in turns on 2026-10-13: each model plays 150 pieces of a seed alone with the filter on, then the next model plays the same seed, for 6 seeds (load average 4.6). A decision is one landing rated: tokenizer, host arrays, the model call and the readout.

| Model | Decisions | Median | Mean | p95 | Decisions/min |
| --- | ---: | ---: | ---: | ---: | ---: |
| d1-omni-600M, int8 weights | 5,094 | 3.62 ms | 3.72 ms | 4.06 ms | 16,179 |
| d1-omni-600M, FP16 weights | 5,094 | 6.06 ms | 6.11 ms | 6.36 ms | 9,800 |
| laya-multilingual, FP16 (their Core ML package) | 5,003 | 4.05 ms | 4.07 ms | 4.23 ms | 14,658 |

## Design checks

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

