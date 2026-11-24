# gpu-forecast-eval

Test stand for independent validation of krauncher forecasts. For each task
it records, before anything runs, the analyzer's assay (the passport: the
work on the reference card) and the GPU ladder (each GPU's compute coefficient
against the reference card), stamped with the UTC time and the analyzer
`calibration_id`, and compares both with the task's reference values.

**The stand measures the quality of Krauncher's forecast; it does not execute
code.** Tasks are never submitted to a GPU: the analyzer reads their source
code and call arguments, no data is downloaded and no GPU time is spent.

## How to run the test

### 1. Requirements

- Python 3.11 or newer, `git`.

### 2. Download

    git clone https://github.com/krauncher/gpu-forecast-eval.git
    cd gpu-forecast-eval
    ./setup.sh

`setup.sh` clones the current krauncher client from git into `client/` and
installs it into a local `.venv`. `./setup.sh <ref>` pins the client to a
branch, tag or commit.

### 3. Get a Krauncher API key

1. Sign up at https://krauncher.com.
2. Create a key: Account → API Keys.
3. Put it into `.env`:

       cp .env.example .env
       # edit .env: KRAUNCHER_API_KEY=cas_...

### 4. Request access to the GPU ladder

The assay is available to every account. The GPU ladder is not publicly
available: access is granted per account on personal request. Write to
admin@krauncher.com with the email of your Krauncher account.

Without ladder access a run stops with
`Ladder access is not enabled for this account.`

### 5. Run

One task:

    .venv/bin/python run.py krauncher_tutorials/bert_imdb

`--service <name>` picks the service under test (default `krauncher`; the
services are adapters in `adapters/`, see `interface.py`).

All tasks:

    for t in tasks/*/*.json; do t=${t#tasks/}; .venv/bin/python run.py "${t%.json}"; done

`resnet152_food101` names the public Food-101 dataset through a data source
`food-101`, from which the analyzer takes the dataset size; `run.py` registers
it on your account if it is missing.

### 6. Read the result

Each run prints reference / service pairs for the passport (`reference_sec`,
`min_vram_gb`) and for every GPU of the ladder (`compute_ratio`), and writes
`results/<service>/<task>_<UTC time>.json`: the forecast normalised to the
stand's fields, the service's answer as received (for Krauncher: assay and
ladder), its version (`calibration_id`, client commit) and the task options
and arguments. Result
files are not edited after they are written: the UTC time and
`calibration_id` show that the forecast was made before any run it is
compared with. `results/` is not tracked by git: the files stay with whoever
runs the stand; keep them, or hand them over together with the measurements.

### 7. Report

    .venv/bin/python report.py [--service krauncher] [--version <calibration_id>] [--out DIR]

Writes `report.json` (machine-readable), `report.md` and `report.html` (for
viewing: every value marked accurate / miss / problem; the top is one summary
over all measurement sets) to `DIR` (default `results/report/`), over the
latest result of every task, per service:
forecast time
(request to normalised forecast, measured by the stand), classification /
assay fields (match per field), reference-card time (centre, typical
deviation, inside the issued spread), compute time on a set's anchor GPU (the GPU's own time: the compute phase
without the host part of the calls, which is the same on every card — what
published benchmarks time),
VRAM (centre, forecasts below the
measured peak) and the ladder (error per GPU, rank correlation of the card
order, time regret of the GPU picked as fastest).

## Tasks

Krauncher client tutorials (`tasks/krauncher_tutorials/`, measured on Krauncher's calibration runs — not independent):

| Module | Task |
|---|---|
| `bert_imdb` | BERT fine-tuning on IMDB |
| `qwen25_7b_lora_alpaca` | Qwen2.5-7B LoRA fine-tuning on Alpaca |
| `qwen25_7b_gsm8k` | Qwen2.5-7B-Instruct inference on GSM8K |
| `qwen25_7b_batched` | Qwen2.5-7B-Instruct batched inference |
| `qwen25_7b_long` | Qwen2.5-7B-Instruct long generation |
| `phi3_inference` | Phi-3-mini inference on GSM8K |
| `qwen15_inference` | Qwen2.5-1.5B-Instruct inference on GSM8K |
| `bert_batch_inference` | BERT batch inference |
| `vit_batch_inference` | ViT batch inference |
| `resnet152_food101` | ResNet-152 training on Food-101 |

Published benchmarks (`tasks/<source>/`, independent measurements):

| Task | Source |
|---|---|
| `lambdalabs/resnet50_amp_bs1280` | ResNet-50 training, AMP, batch 1280, synthetic data; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/resnet50_amp_bs928` | ResNet-50 training, AMP, batch 928, synthetic data; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/resnet50_amp_bs448` | ResNet-50 training, AMP, batch 448, synthetic data; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/resnet50_fp32_bs224` | ResNet-50 training, fp32, batch 224, synthetic data; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/resnet50_fp32_bs496` | ResNet-50 training, fp32, batch 496, synthetic data; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, 48 GB) |
| `lambdalabs/resnet50_fp32_bs640` | ResNet-50 training, fp32, batch 640, synthetic data; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/resnet50_amp_bs720` | ResNet-50 training, AMP, batch 720, synthetic data; Lambda GPU benchmark (A100 40GB PCIe) |
| `lambdalabs/resnet50_amp_bs2560` | ResNet-50 training, AMP, batch 2560, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (H200 141 GB) |
| `lambdalabs/resnet50_amp_bs1360` | ResNet-50 training, AMP, batch 1360, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (GH200 96 GB) |
| `lambdalabs/resnet50_amp_bs3328` | ResNet-50 training, AMP, batch 3328, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (B200 192 GB) |
| `lambdalabs/resnet50_fp32_bs1280` | ResNet-50 training, fp32, batch 1280, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (H200 141 GB) |
| `lambdalabs/resnet50_fp32_bs720` | ResNet-50 training, fp32, batch 720, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (GH200 96 GB) |
| `lambdalabs/resnet50_fp32_bs1664` | ResNet-50 training, fp32, batch 1664, synthetic data; Lambda GPU benchmark v2, NGC 24.10 (B200 192 GB) |
| `timm/resnet152_amp_bs192` | ResNet-152 training step benchmark, AMP, batch 192, synthetic input; timm benchmark.py results (RTX 3090) |
| `lambdalabs/bert_base_squad_amp_bs320` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 320, 101 steps; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/bert_base_squad_amp_bs192` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 192, 101 steps; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/bert_base_squad_amp_bs96` | BERT-base fine-tuning on SQuAD v1.1, mixed precision, batch 96, 101 steps; Lambda GPU benchmark (RTX 4090, RTX 3090, 24 GB) |
| `lambdalabs/bert_large_squad_amp_bs112` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 112, 101 steps; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/bert_large_squad_amp_bs64` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 64, 101 steps; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/ncf_ml20m_bs10m` | NCF (NeuMF) training on MovieLens-20M, batch 10 000 000, 2 epochs, fp32; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/ncf_ml20m_amp_bs10m` | NCF (NeuMF) training on MovieLens-20M, mixed precision (apex amp O2), batch 10 000 000, 2 epochs; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000, 48 GB) |
| `lambdalabs/ncf_ml20m_bs4278184` | NCF (NeuMF) training on MovieLens-20M, mixed precision (apex amp O2), batch 4 278 184, 2 epochs; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/waveglow_ljs625_bs48` | WaveGlow training on LJSpeech (625 clips, 8000-sample segments), batch 48, 2 epochs, fp32; model and data code verbatim from NVIDIA DeepLearningExamples (BSD 3-Clause); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/waveglow_ljs625_bs32` | the same WaveGlow training, batch 32; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000) |
| `lambdalabs/waveglow_ljs625_bs18` | the same WaveGlow training, batch 18; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `lambdalabs/tacotron2_ljs625_bs256` | Tacotron 2 training on LJSpeech (625 clips), batch 256, 3 epochs, fp32; model, data and text code verbatim from NVIDIA DeepLearningExamples (BSD 3-Clause; text front end MIT, Apache 2.0); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/tacotron2_ljs625_bs148` | the same Tacotron 2 training, batch 148, 1 epoch; Lambda GPU benchmark (RTX 6000 Ada, RTX A6000, Quadro RTX 8000) |
| `lambdalabs/tacotron2_ljs625_bs88` | the same Tacotron 2 training, batch 88, 2 epochs; Lambda GPU benchmark (RTX 4090, RTX 3090) |
| `lambdalabs/transformer_xl_base_wt103_bs104` | Transformer-XL base training on WikiText-103, batch 104, 400 steps, fp16; two phases (data preparation, training); model, optimizer and data code verbatim from NVIDIA DeepLearningExamples (Apache 2.0); Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/transformer_xl_base_wt103_bs64`, `transformer_xl_base_wt103_bs24` | the same, batch 64, 40 steps (RTX 6000 Ada, RTX A6000, Quadro RTX 8000); batch 24, 400 steps (RTX 4090, RTX 3090, A10) |
| `lambdalabs/transformer_xl_large_wt103_bs48` | Transformer-XL large training on WikiText-103, batch 48, 400 steps, fp16; Lambda GPU benchmark (A100 / H100 80 GB) |
| `lambdalabs/transformer_xl_large_wt103_bs32`, `transformer_xl_large_wt103_bs8` | the same, batch 32, 40 steps (RTX 6000 Ada, RTX A6000, Quadro RTX 8000); batch 8, 400 steps (RTX 4090, RTX 3090, A10) |
| `lambdalabs/bert_large_squad_amp_bs28` | BERT-large fine-tuning on SQuAD v1.1, mixed precision, batch 28, 101 steps; Lambda GPU benchmark (RTX 4090, RTX 3090, A10, 24 GB) |
| `tensorrt_llm/llama31_8b_fp8_isl1000_osl1000`, `_isl128_osl2048`, `_isl2048_osl128` | TensorRT-LLM offline throughput, Llama 3.1 8B FP8, 3000 requests per ISL/OSL; NVIDIA perf overview (H100, H200, GH200) |
| `tensorrt_llm/llama33_70b_fp8_tp2_isl1000_osl1000` | the same, Llama 3.3 70B FP8, TP 2 (H100, H200) |
| `wan_video/wan21_t2v_14b_720p`, `wan21_i2v_14b_720p` | Wan2.1 text- / image-to-video 14B, 720p; Wan2.1 README efficiency table (A100, H100) |
| `wan_video/wan21_t2v_1_3b_480p` | Wan2.1 text-to-video 1.3B, 480p, one RTX 4090; the same table |
| `cogvideo/cogvideox_5b_t2v` | CogVideoX-5B text-to-video, 720x480, 49 frames, 50 steps, BF16, diffusers; THUDM/CogVideo README (A100, H100) |
| `cogvideo/cogvideox_2b_t2v`, `cogvideox15_5b_t2v` | CogVideoX-2B (FP16) and CogVideoX1.5-5B (BF16) text-to-video, diffusers; the same README (A100, H100) |
| `tensorrt_llm/llama31_8b_fp8_isl128_osl128`, `_isl128_osl4096`, `_isl500_osl2000`, `_isl1000_osl2000`, `_isl2048_osl2048`, `_isl5000_osl500`, `_isl20000_osl2000` | the same TensorRT-LLM offline throughput, Llama 3.1 8B FP8, TP 1, further ISL/OSL rows of the perf overview (H100, H200, GH200) |
| `wan_video/wan22_t2v_a14b_480p`, `wan22_t2v_a14b_720p` | Wan2.2 text-to-video A14B, 480p / 720p; Wan2.2 README efficiency table (A100, H100) |
| `wan_video/wan22_ti2v_5b_t2v_720p` | Wan2.2 TI2V-5B text-to-video, 720p, one RTX 4090; the same table |
| `pytorch_blog/sdxl_bf16_a100` | SDXL text-to-image, BF16, SDPA; PyTorch blog "Accelerating Generative AI Part III: Diffusion, Fast" (A100 80 GB) |
| `runcrate/sdxl_fp16_bs1` | SDXL text-to-image, FP16, eager, batch 1; Runcrate "SDXL images per dollar on four GPUs" (RTX A6000, L40S, RTX 5090, H100 PCIe) |

## Adding a task

The procedure, for your own workload or a published benchmark:
[doc/adding_samples.md](doc/adding_samples.md).

A task is its neutral code `tasks/<name>.py` — a plain script runnable on a
GPU machine (the task function, the call arguments `KWARGS`, models and
datasets by public ids) — and its description `tasks/<name>.json`: the
expected classification and assay fields (read from the code) and one or more
measurement sets (`reference_sec` or `anchor_compute_sec`, `min_vram_gb` or
`vram_gb_at_most`, `compute_ratio` per GPU), each
with its source, date and whether it is independent of the service's
calibration. A service takes the task in its own form, a
prepared file `variants/<service>/<name>.py`. The Krauncher variant defines
the task function (`FUNC`) and the arguments of its `@client.task` decorator
(`OPTIONS`) and of the call (`KWARGS`); a task that reads a registered data
source (`OPTIONS["data"]`) lists it in `DATA_SOURCES`, and the run registers
it on the account when missing. Where every expected `steps` /
`dataset_samples` value comes from: [doc/reference_values.md](doc/reference_values.md).
