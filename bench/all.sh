#!/bin/zsh
# Every benchmark behind RESULTS.md, one engine at a time (nothing else should be using the ANE or the GPU).
# Results: bench/results/*.json. A run whose output file exists is skipped; delete the file to redo it.
#
#   uv sync --extra bench
#   uv run d1ane setup --laya                                              # the default build (int8 weights) and laya
#   uv run d1ane setup --quant fp16 --build build/d1-omni-600M-fp16        # the FP16 reference build
#   bench/all.sh && uv run python bench/report.py > RESULTS.md
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
INT8=build/d1-omni-600M
FP16=build/d1-omni-600M-fp16
R=bench/results
py() {
  [ -f "${@[-1]}" ] && { echo "have ${@[-1]}"; return; }
  $PY -I -W ignore "$@" 2>&1 | grep -v "Warning\|warn\|^W[0-9]\|Loading weights\|Torch version"
}
get() { [ -f "bench/data/$1/$3" ] || $PY -c "from huggingface_hub import hf_hub_download as d; d('$2', '$3', repo_type='dataset', local_dir='bench/data/$1')" >/dev/null; }
mkdir -p bench/data $R
[ -f bench/data/cats.jpg ] || curl -sL -o bench/data/cats.jpg http://images.cocodataset.org/val2017/000000039769.jpg
[ -f bench/data/speech.flac ] || curl -sL -o bench/data/speech.flac https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac
for c in random popular adversarial; do get pope lmms-lab/POPE Full/$c-00000-of-00001.parquet; done
get slurp DynamicSuperb/SuperbIC_SLURP-Intent data/test-00000-of-00001.parquet
get minds14 PolyAI/minds14 en-US/train-00000-of-00001.parquet

# text
py bench/run.py latency --engine coreai-ane     --build $FP16 --reps 30 --warmup 5 --out $R/latency_coreai-ane.json
py bench/run.py latency --engine coreai-gpu     --build $FP16 --reps 15 --warmup 3 --out $R/latency_coreai-gpu.json
py bench/run.py latency --engine torch-mps-fp16 --reps 15 --warmup 3 --out $R/latency_torch-mps-fp16.json
py bench/run.py dataset --engine torch-mps-fp32 --out $R/dataset_torch-mps-fp32.json    # the FP32 reference
py bench/run.py dataset --engine coreai-ane     --build $FP16 --out $R/dataset_coreai-ane.json
py bench/run.py dataset --engine coreai-ane     --build $INT8 --out $R/dataset_coreai-ane-int8.json
py bench/run.py dataset --engine coreai-gpu     --build $FP16 --out $R/dataset_coreai-gpu.json
py bench/run.py dataset --engine torch-mps-fp16 --out $R/dataset_torch-mps-fp16.json
py bench/quant_latency.py --fp16 $FP16 --int8 $INT8 --out $R/quant_ane.json

# images
py bench/run.py vision --engine coreai-ane     --build $FP16 --reps 15 --warmup 3 --out $R/vision_coreai-ane.json
py bench/run.py vision --engine coreai-gpu     --build $FP16 --reps 15 --warmup 3 --out $R/vision_coreai-gpu.json
py bench/run.py vision --engine torch-mps-fp16 --reps 15 --warmup 3 --out $R/vision_torch-mps-fp16.json
py bench/run.py pope   --engine torch-mps-fp32 --out $R/pope_torch-mps-fp32.json        # the FP32 reference
py bench/run.py pope   --engine coreai-ane     --build $FP16 --out $R/pope_coreai-ane.json
py bench/run.py pope   --engine coreai-ane     --build $INT8 --out $R/pope_coreai-ane-int8.json
py bench/run.py pope   --engine torch-mps-fp16 --out $R/pope_torch-mps-fp16.json

# audio
py bench/run.py audio  --engine coreai-ane     --build $FP16 --reps 15 --warmup 3 --out $R/audio_coreai-ane.json
py bench/run.py audio  --engine coreai-gpu     --build $FP16 --reps 15 --warmup 3 --out $R/audio_coreai-gpu.json
py bench/run.py audio  --engine torch-mps-fp16 --reps 15 --warmup 3 --out $R/audio_torch-mps-fp16.json
py bench/run.py speech --engine torch-mps-fp32 --out $R/speech_torch-mps-fp32.json      # the FP32 reference
py bench/run.py speech --engine coreai-ane     --build $FP16 --out $R/speech_coreai-ane.json
py bench/run.py speech --engine coreai-ane     --build $INT8 --out $R/speech_coreai-ane-int8.json
py bench/run.py speech --engine torch-mps-fp16 --out $R/speech_torch-mps-fp16.json

# Tetris: each model with the wording and question tuned to it, then swapped, then fresh seeds
py bench/tetris.py --policy d1   --build $INT8 --shortlist --out $R/tetris_d1-int8_calibrated-rating-shortlist.json
py bench/tetris.py --policy d1   --build $INT8             --out $R/tetris_d1-int8_calibrated-rating.json
py bench/tetris.py --policy laya --style graded --shortlist --out $R/tetris_laya_graded-shortlist.json
py bench/tetris.py --policy laya --style graded            --out $R/tetris_laya_graded.json
py bench/tetris.py --policy laya --style plain --shortlist --out $R/tetris_laya_plain-shortlist.json
py bench/tetris.py --policy laya --style plain             --out $R/tetris_laya_plain.json
for style in plain graded; do
  py bench/tetris.py --policy d1 --build $INT8 --style $style --ask clean --shortlist --out $R/tetris_d1-int8_$style-shortlist.json
  py bench/tetris.py --policy d1 --build $INT8 --style $style --ask clean             --out $R/tetris_d1-int8_$style.json
done
py bench/tetris.py --policy laya --style calibrated --ask rating --shortlist --out $R/tetris_laya_calibrated-rating-shortlist.json
py bench/tetris.py --policy laya --style calibrated --ask clean  --shortlist --out $R/tetris_laya_calibrated-clean-shortlist.json
py bench/tetris.py --policy d1   --build $INT8 --shortlist --seeds 11-20 --out $R/tetris_d1-int8_calibrated-rating-shortlist_seeds11-20.json
py bench/tetris.py --policy laya --style graded --shortlist --seeds 11-20 --out $R/tetris_laya_graded-shortlist_seeds11-20.json
py bench/tetris_speed.py --build $INT8 --fp16 $FP16 --out $R/tetris_speed.json
