# DLE Tacotron 2 file lists

`ljs_audio_text_train_subset_625_filelist.txt` — the training file list of the
Lambda Tacotron 2 and WaveGlow benchmarks (`tasks/lambdalabs/tacotron2_*`,
`waveglow_*`), copied unchanged from github.com/LambdaLabsML/DeepLearningExamples,
commit `667536cc8138edbd236aed08dbafc8e77bcc7c20`,
`PyTorch/SpeechSynthesis/Tacotron2/filelists/` (the same file as the URL in the
tasks' data source). 625 lines, one clip each (`path|text`).

The Krauncher variants count it with `krauncher.DataManifest.from_file` and pass
the result as the `train_list` argument: the analyzer sees the task's code, not
its data, and the file list sets the size of the training set.

Licence: BSD 3-Clause, NVIDIA Corporation — `LICENCE`, copied from
`PyTorch/SpeechSynthesis/Tacotron2/LICENCE` at the same commit.
