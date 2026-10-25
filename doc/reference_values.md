# Reference values: `steps` and `dataset_samples`

Where each expected value of these two fields in `tasks/*/*.json` comes from,
what it rests on, and how it was checked, so that each one can be re-checked.
Values set on 2026-10-25. Code references are to the reference form
`tasks/<source>/<name>.py` (the `variants/` form a service receives runs the
same loops); line numbers are from that date, the quoted code is what to search
for if they drift.

## Definitions

- `steps` — batch passes of the run: training micro-batches (forward and
  backward of one batch), forward or `generate` calls in inference.
  Evaluation and validation passes are not counted. Under gradient accumulation
  the micro-batches are counted, not the optimizer steps.
- `dataset_samples` — the set the loop actually iterates over in one epoch
  (or in the run, without epochs): rows, examples, clips, or the items the
  code builds from them (tokenized features, sampled negatives), before
  batching. Not the number processed — a loop cut short (`max_batches`,
  `max_steps`) is counted in `steps`. Decided 2026-10-25.

"Checked by": **code** — read from the task code and its call arguments
(`KWARGS`); **source** — an external number confirmed in the cited source on
2026-10-25; **calculation** — computed by running the task's own data
preparation off the GPU; **run** — counted in an actual run. No value is checked by a run
yet.

## Values

### krauncher_tutorials

| task | field | value | derivation | premises (source) | checked by |
|---|---|---|---|---|---|
| `resnet152_food101` | steps | 450 | `KWARGS` epochs 3, max_batches 150; `n_batches = ... min(max_batches, len(train_loader))` (l. 59); `if i > n_batches: break` with `enumerate(train_loader, 1)` (l. 76-78) runs 150 batches per epoch | `len(train_loader)` >= 150, i.e. >= 9 600 images: the loader holds 101 000 / 64 = 1 579 batches | code |
| `resnet152_food101` | dataset_samples | 101000 | `ImageFolder(os.path.join(data_root, "images"))` (l. 54): every image of the archive, no split filter. The comment "Build train split from meta/train.txt" (l. 46) is not what the code does | the archive holds all of Food-101: 101 categories, 101 000 images, 750 train + 250 test per class (data.vision.ee.ethz.ch/cvl/datasets_extra/food-101) | code, source |
| `bert_imdb` | steps | 4689 | HF `Trainer`, `num_train_epochs` 3, `per_device_train_batch_size` 16 (l. 57-58): 3 x ceil(25 000 / 16) = 3 x 1 563. `eval_strategy="epoch"` (l. 62) adds evaluation passes, not counted | the last partial batch is kept: `TrainingArguments.dataloader_drop_last` defaults to False (transformers `src/transformers/training_args.py`); IMDB train split = 25 000 rows (datasets-server, `stanfordnlp/imdb`) | code, source |
| `bert_imdb` | dataset_samples | 25000 | `train_dataset = ds["train"]` (l. 49) | IMDB train = 25 000 (datasets-server, `stanfordnlp/imdb`) | source |
| `qwen25_7b_lora_alpaca` | steps | 2250 | `Trainer`, `num_epochs` 3, `batch_size` 4, `grad_accum` 4 (defaults l. 13-16): 3 x ceil(3 000 / 4) = 3 x 750 micro-batches (optimizer steps would be ~188 per epoch) | 3 000 rows selected: `select(range(min(num_samples, len(ds_full))))` (l. 81) with Alpaca train = 52 002 rows (datasets-server, `tatsu-lab/alpaca`); `dataloader_drop_last` False as above | code, source |
| `qwen25_7b_lora_alpaca` | dataset_samples | 3000 | as above: min(3 000, 52 002) | as above | code, source |
| `phi3_inference`, `qwen15_inference`, `qwen25_7b_batched`, `qwen25_7b_gsm8k`, `qwen25_7b_long` | steps | 50, 80, 30, 200, 30 | one `model.generate` per sample of `ds.select(range(min(num_samples, len(ds_full))))`; `num_samples` defaults 50 / 80 / 30 / 200 / 30 | GSM8K `main` test split = 1 319 rows >= 200 (datasets-server, `openai/gsm8k`) | code, source |
| same five | dataset_samples | 50, 80, 30, 200, 30 | as above | as above | code, source |
| `bert_batch_inference`, `vit_batch_inference` | steps | 1 | one `model(**inputs)` over one batch built in the code, no loop (bert l. 51-52) | — | code |

### lambdalabs

| task | field | value | derivation | premises (source) | checked by |
|---|---|---|---|---|---|
| `bert_{base,large}_squad_amp_bs*` (6) | steps | 101 | `step, t0 = 0, ...; while step <= max_steps` (l. 100-101), `max_steps` 100: steps 0..100. The source does the same (run_squad.py:1060-1061, cited in the task docstring) | — | code |
| same six | dataset_samples | 88492 | the loop iterates `TensorDataset` of the features `tokenizer(ds["question"], ds["context"], truncation="only_second", max_length=384, stride=128, return_overflowing_tokens=True, ...)` (l. 69-71): 88 492 features from 87 599 examples | SQuAD train = 87 599 rows (datasets-server, `rajpurkar/squad`); features counted 2026-10-25 with that call on the official `train-v1.1.json` (rajpurkar.github.io/SQuAD-explorer) and the tokenizers of `google-bert/bert-base-uncased` and `bert-large-uncased` (the same vocabulary, the same count); the premise is that the HF dataset holds the same question / context pairs as the JSON | code, source, calculation |
| `ncf_ml20m_amp_bs10m`, `ncf_ml20m_bs10m` | steps | 18 | 2 epochs x 9: per epoch 19 861 770 positives + 4 negatives each = 99 308 850 samples, `order.split(batch_size)[:-1]` -> 10 chunks of 10 000 000, the last (partial) one dropped | positives as in dataset_samples below | code, source |
| `ncf_ml20m_bs4278184` | steps | 46 | 2 epochs x 23: 99 308 850 / 4 278 184 = 23.2 -> 24 chunks, last dropped | as above | code, source |
| all three NCF | dataset_samples | 99308850 | per epoch the loop iterates `order = torch.randperm(len(users))` over the positives plus 4 sampled negatives each: positives = `ratings.csv` rows minus the held-out test (`test = df.groupby("userId").tail(1)`; `train = df.drop(test.index)`) = 20 000 263 - 138 493 = 19 861 770; x (1 + 4) = 99 308 850 | ML-20M: 20 000 263 ratings, 138 493 users, every user rated at least 20 movies, so each user gives exactly one test row (files.grouplens.org/datasets/movielens/ml-20m-README.html) | code, source |
| `resnet50_amp_bs1280`, `_bs448`, `_bs928` | steps | 200, 200, 50 | `for epoch in range(epochs): for i in range(iterations_per_epoch)` (l. 66-67); `KWARGS` 2 x 100, 2 x 100, 1 x 50 | — (synthetic batch, no dataset_samples) | code |
| `tacotron2_ljs625_bs148`, `_bs256`, `_bs88` | steps | 4, 6, 14 | epochs x floor(625 / batch), `drop_last=True` (l. 1672): 1 x 4, 3 x 2, 2 x 7 | the file list has 625 entries and the dataset keeps all of them (`__len__` = `len(self.audiopaths_and_text)`, l. 1545-1546) | code, source |
| `waveglow_ljs625_bs18`, `_bs32`, `_bs48` | steps | 68, 38, 26 | 2 epochs x floor(625 / batch), `drop_last=True` (l. 884): 2 x 34, 2 x 19, 2 x 13 | as for Tacotron 2; clips shorter than the segment are padded, not dropped (`get_mel_audio_pair`) | code, source |
| all six LJSpeech tasks | dataset_samples | 625 | lines of `ljs_audio_text_train_subset_625_filelist.txt`, read by `load_filepaths_and_text` | downloaded from the URL in the variant's `data_urls` (LambdaLabsML/DeepLearningExamples `667536cc`): 625 lines, 625 distinct paths, every line `path\|text` | source |
| `transformer_xl_{base,large}_wt103_bs*` (6) | steps | 400 or 40 (= `max_step`) | `for epoch in itertools.count(start=1)` with `if train_step == max_step: break` (base_bs104 l. 2170-2173): exactly `max_step` steps, however many epochs that takes | — (no dataset_samples) | code |

### tensorrt_llm

| task | field | value | derivation | premises (source) | checked by |
|---|---|---|---|---|---|
| `llama31_8b_fp8_*` (3), `llama33_70b_fp8_tp2_isl1000_osl1000` | dataset_samples | 3000 | `prompts = [... for _ in range(num_requests)]`, `KWARGS` num_requests 3000, passed whole to `llm.generate(prompts, params)`; the warm-up `llm.generate(prompts[:warmup], params)` runs on a slice of the same list. `batch_size` is null: the engine forms the batch of every step from the requests in flight (continuous batching), bounded by the KV cache and max_batch_size, so it depends on the GPU, not on the code | 3000 requests for these ISL/OSL in the source's dataset table (TensorRT-LLM `perf-overview.md` at `0c9430e`, lines 215-226) | code, source |

## Open points

1. BERT: `steps` is 101, while `anchor_compute_sec_note` in the same task
   files computes the work from 100 steps (19 200 sequences for bs192).
2. No value is confirmed by a run. Of the tasks, only ResNet-50 returns its
   step count (`steps = epochs * iterations_per_epoch`); a printed step counter
   in each task would turn "code" into "run".
