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

Notes:
- transformers removed `examples/` from main on 2026-10-13 (`14d7637051`,
  "Remove examples"); the checkout is its parent, the last state with them.
- `mlcommons/inference_results_v5.0` has no licence file; v6.0 (Apache-2.0)
  taken instead.

## Exceptions inside the trees

| Path | Licence | stand | training |
|---|---|---|---|
| `mlcommons__training/retired_benchmarks/gpt3/megatron-lm/tasks/orqa/unsupervised/{qa_utils,tokenizers}.py` | CC-BY-NC 4.0 (taken from facebookresearch/DPR) | yes (non-commercial) | **no** |
| `mlcommons__training/retired_benchmarks/stable_diffusion/` | CreativeML Open RAIL-M (use-based restrictions) | yes | check before use |
| `mlcommons__training/retired_benchmarks/gpt3/megatron-lm/` | BSD-3 + Apache-2.0 (Megatron-LM) | yes | yes |
| `CentML__DeepView.Predict/experiments/{dcgan,inception,resnet}/` | BSD-3 | yes | yes |
| `CentML__DeepView.Predict/experiments/{gnmt/seq2seq,transformer}/` | MIT | yes | yes |
| `mlcommons__*/retired_benchmarks/**/rnnt/**/text/`, `retinanet/ssd/`, `gnmt/pytorch/`, minigo UI | MIT / BSD | yes | yes |

Not covered: licences of the models and datasets the scripts load (each
Hugging Face model and dataset has its own; some MLPerf datasets are gated).
These are checked per task when a task is built.

## Not downloaded

- Cineca users' workloads — need an agreement (via Laura).
- Hugging Face Hub model configs — per-model licences; the registry sync
  (`cas-analyzer/tools/hf_registry_sync.py`) already reads them.
