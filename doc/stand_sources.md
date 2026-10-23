# Stand sources — raw downloads (collection stage)

Downloaded 2026-10-20 for the validation stand (`gpu-forecast-eval`), sources
other than `lambdalabs`, into `/home/cas/src/stand_sources/<folder>` (outside this
repository). Nothing here is used by the stand yet. Shallow clones;
code is not executed.

Licence roles:
- **stand** — use allowed at least for non-commercial evaluation.
- **training** — the licence allows using the code to train the analyzer
  (permissive: Apache-2.0, MIT, BSD).

## Sources

| Folder | Upstream | Commit | Date | Repo licence | stand | training |
|---|---|---|---|---|---|---|
| `mlcommons__training` | github.com/mlcommons/training | `7bf4fbd2b8cb` | 2026-09-28 | Apache-2.0 | yes | yes, except below |
| `mlcommons__inference` | github.com/mlcommons/inference | `3fbc32993999` | 2026-09-02 | Apache-2.0 | yes | yes, except below |
| `mlcommons__training_results_v6.0` | github.com/mlcommons/training_results_v6.0 | `eabf23a07b2a` | 2026-06-15 | Apache-2.0 (submitters' files may carry own headers) | yes | per file header |
| `mlcommons__inference_results_v6.0` | github.com/mlcommons/inference_results_v6.0 | `4d3916ac9cf4` | 2026-04-03 | Apache-2.0 (submitters' files may carry own headers) | yes | per file header |
| `huggingface__transformers` (`examples/` only) | github.com/huggingface/transformers | `cc0618a18035` | 2026-10-13 | Apache-2.0 | yes | yes |
| `huggingface__diffusers` (`examples/` only) | github.com/huggingface/diffusers | `e04f0c5f2439` | 2026-10-19 | Apache-2.0 | yes | yes |
| `CentML__DeepView.Predict` | github.com/CentML/DeepView.Predict (archived) | `8f8e682b0738` | 2024-12-16 | Apache-2.0 | yes | yes |
| `eriklindernoren__PyTorch-GAN` | github.com/eriklindernoren/PyTorch-GAN | `36d3c77` | 2021-01-06 | MIT | yes | yes |
| `junyanz__pytorch-CycleGAN-and-pix2pix` | github.com/junyanz/pytorch-CycleGAN-and-pix2pix | `2a7afba` | 2025-08-06 | BSD (CycleGAN, pix2pix, DCGAN parts) | yes | yes |
| `NVIDIA__pix2pixHD` | github.com/NVIDIA/pix2pixHD | `14b3b3c` | 2024-09-30 | BSD | yes | yes |
| `lucidrains__lightweight-gan` | github.com/lucidrains/lightweight-gan | `efbec45` | 2025-01-11 | MIT | yes | yes |
| `ajbrock__BigGAN-PyTorch` | github.com/ajbrock/BigGAN-PyTorch | `9845943` | 2019-07-18 | MIT | yes | yes |
| `rosinality__stylegan2-pytorch` | github.com/rosinality/stylegan2-pytorch | `bef283a` | 2021-11-24 | MIT + NVIDIA Source Code License (see exceptions) | yes | **no** |
| `NVlabs__stylegan2-ada-pytorch` | github.com/NVlabs/stylegan2-ada-pytorch | `d72cc7d` | 2023-09-22 | NVIDIA Source Code License (non-commercial: research or evaluation only) | yes | **no** |
| `NVlabs__stylegan3` | github.com/NVlabs/stylegan3 | `c233a91` | 2023-04-26 | NVIDIA Source Code License (non-commercial: research or evaluation only) | yes | **no** |
| `mlcommons__inference_results_v5.1` (`summary_results.json` only) | github.com/mlcommons/inference_results_v5.1 | `5ea4f62` | 2026-01-16 | Apache-2.0 | yes | per file header |
| `Stability-AI__generative-models` (`scripts/`, `sgm/`, `configs/`) | github.com/Stability-AI/generative-models | `e8cd657` | 2025-12-16 | MIT (SDXL weights: CreativeML Open RAIL++-M) | yes | yes |
| `facebookresearch__detectron2` (`demo/`, `tools/`, `configs/`) | github.com/facebookresearch/detectron2 | `1e3e13b` | 2026-09-29 | Apache-2.0 | yes | yes |
| `open-mmlab__mmdetection` (`demo/`, `tools/`, RetinaNet / YOLOX configs) | github.com/open-mmlab/mmdetection | `cfd5d3a` | 2024-02-05 | Apache-2.0 | yes | yes |
| `MIC-DKFZ__nnUNet` (`nnunetv2/inference/`) | github.com/MIC-DKFZ/nnUNet | `bbf1222` | 2026-10-16 | Apache-2.0 | yes | yes |
| `Project-MONAI__tutorials` (`3d_segmentation/`) | github.com/Project-MONAI/tutorials | `f33710c` | 2026-10-15 | Apache-2.0 | yes | yes |
| `vllm-project__vllm` (`benchmarks/`, `examples/basic/offline_inference/`) | github.com/vllm-project/vllm | `e4a676c` | 2026-10-20 | Apache-2.0 | yes | yes |
| `sgl-project__sglang` (`python/sglang/benchmark/`, `examples/runtime/engine/`, `benchmark/gsm8k/`) | github.com/sgl-project/sglang | `f8fd41c` | 2026-10-20 | Apache-2.0 | yes | yes |

Notes:
- transformers removed `examples/` from main on 2026-10-13 (`14d7637051`,
  "Remove examples"); the checkout is its parent, the last state with them.
- `mlcommons/inference_results_v5.0` has no licence file; v6.0 (Apache-2.0)
  taken instead.
- `inference_results_v5.1`: only the summary table, for the single-accelerator
  results v6.0 lacks (Stable Diffusion XL, RetinaNet, 3D-UNet KiTS19).
- Inference sources added 2026-10-20 as tree-side partners of the MLPerf
  inference tasks. Not taken: `ultralytics/ultralytics` (AGPL-3.0, the YOLO
  task's run-time dependency) and `Comfy-Org/ComfyUI` (GPL-3.0).
- GAN sources added 2026-10-20 (the stand's GAN class had one family, DeepView
  dcgan). `pytorch/examples` dcgan not taken: DeepView's dcgan derives from it
  (same family).

## Exceptions inside the trees

| Path | Licence | stand | training |
|---|---|---|---|
| `mlcommons__training/retired_benchmarks/gpt3/megatron-lm/tasks/orqa/unsupervised/{qa_utils,tokenizers}.py` | CC-BY-NC 4.0 (taken from facebookresearch/DPR) | yes (non-commercial) | **no** |
| `mlcommons__training/retired_benchmarks/stable_diffusion/` | CreativeML Open RAIL-M (use-based restrictions) | yes | check before use |
| `mlcommons__training/retired_benchmarks/gpt3/megatron-lm/` | BSD-3 + Apache-2.0 (Megatron-LM) | yes | yes |
| `CentML__DeepView.Predict/experiments/{dcgan,inception,resnet}/` | BSD-3 | yes | yes |
| `CentML__DeepView.Predict/experiments/{gnmt/seq2seq,transformer}/` | MIT | yes | yes |
| `rosinality__stylegan2-pytorch/op/` and the model details of `model.py` | NVIDIA Source Code License (from NVlabs/stylegan2; `LICENSE-NVIDIA`); `lpips/` BSD, FID code Apache-2.0 | yes | **no** for the repository (training scripts import `op/`) |
| `mlcommons__*/retired_benchmarks/**/rnnt/**/text/`, `retinanet/ssd/`, `gnmt/pytorch/`, minigo UI | MIT / BSD | yes | yes |

Not covered: licences of the models and datasets the scripts load (each
Hugging Face model and dataset has its own; some MLPerf datasets are gated).
These are checked per task when a task is built.

## Not downloaded

- Cineca users' workloads — need an agreement (via Laura).
- Hugging Face Hub model configs — per-model licences; the registry sync
  (`cas-analyzer/tools/hf_registry_sync.py`) already reads them.

## Candidates with published measurements: llm_inference and diffusion_inference (2026-10-23)

Pilot directions of the Laura project. Not downloaded yet; each becomes a task
under `tasks/` after its script is reduced and checked. What the stand compares:
our forecast on the cards the source measured, through `anchor_gpu` and
`compute_ratio`; the reference card need not be among them. Serving under a
request stream is out of the stand's scope.

| Candidate | Source | Published measurements | GPUs | Licence (to check per task) |
|---|---|---|---|---|
| Wan2.1 T2V-14B, 720p, 50 steps | github.com/Wan-Video/Wan2.1, README, `assets/comp_effic.png` | Total time (s) / peak memory (GB), 1/2/4/8 GPUs. 1 GPU: H800/H100 1837.9 / 69.1, A800/A100 3342.6 / 69.1, H20 6935.5 / 69.1 | H100, A100, H20 | Wan2.1 repo and weights — check |
| Wan2.1 I2V-14B, 720p, 40 steps | same table | 1 GPU: H800/H100 1491.6 / 76.7, A800/A100 2715.5 / 76.7, H20 5494.8 / 76.7 | H100, A100, H20 | same |
| Wan2.1 T2V-1.3B, 480p | same table | RTX 4090: 1 GPU 261.4 / 8.19; 2/4/8 GPUs 188.5, 120.8, 112.3 s | RTX 4090 | same |
| CogVideoX-5B, 720×480, 49 frames, 50 steps, BF16 | github.com/THUDM/CogVideo, README | A100 ~180 s, H100 ~90 s (rounded); single-GPU memory from 5 GB with diffusers optimisations | A100, H100 | code Apache-2.0; CogVideoX-5B weights licence — check |
| Llama 3.1 8B Instruct FP8, fixed ISL/OSL (e.g. 128/128, 1000/1000, 2048/2048) | TensorRT-LLM Performance Overview, nvidia.github.io/TensorRT-LLM/performance/perf-overview.html (commit 0c9430e, updated 2025-09-15) | Total output throughput (tokens/s), `trtllm-bench … throughput --backend pytorch`, v0.21, synthetic requests of one length | H100, H200, GH200, B200, GB200 | TensorRT-LLM Apache-2.0; Llama 3.1 Community Licence (gated) |
| Llama 3.3 70B Instruct FP8, multi-GPU | same page | same | same | TensorRT-LLM Apache-2.0; Llama 3.3 Community Licence (gated) |

Notes:
- Wan2.1: "Total time" — whether it includes model loading is to be checked
  before `anchor_compute_sec` is set; peak memory gives `min_vram_gb`.
- TensorRT-LLM: work = number of requests × OSL tokens, so
  `anchor_compute_sec` = work / throughput. Column alignment of the page's
  tables to be checked against the page itself.
- Not taken: HunyuanVideo (the README does not name the GPU of its timing
  table); SemiAnalysis InferenceX (very large multi-node models only).
- MLPerf SDXL: from `inference_results_v5.1` summary, already listed above.
