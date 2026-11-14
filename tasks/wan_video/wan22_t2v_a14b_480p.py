"""Wan2.2 text-to-video A14B (two 14B experts), 832x480, 81 frames, 40 steps, one GPU
with model offload and bf16 weights (Wan-Video/Wan2.2, computational-efficiency table).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/wan_video/wan22_t2v_a14b_480p.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
wan22_t2v_a14b_480p.json; the form a service takes is in
variants/<service>/wan_video/wan22_t2v_a14b_480p.py.

Source (the measured workload):
- github.com/Wan-Video/Wan2.2, commit 1ea34ff (2026-09-21), README.md
  "Computational Efficiency on Different GPUs" (lines 426-440) and
  assets/comp_effic.png, row A100/A800 and H100/H800 / T2V-A14B / 480P, column "1" GPU.
- Command (README.md:147, size 832*480 (SUPPORTED_SIZES['t2v-A14B'], wan/configs/__init__.py:44); the table's single-GPU settings at :436 —
  `--offload_model True --convert_model_dtype`, no prompt extension):
  python generate.py --task t2v-A14B --size 832*480 --ckpt_dir ./Wan2.2-T2V-A14B
      --offload_model True --convert_model_dtype --prompt "Two anthropomorphic
      cats in comfy boxing gear and bright gloves fight intensely on a
      spotlighted stage."

This file calls the same pipeline the command runs; every value follows the
source (paths relative to the repository at 1ea34ff):
- pipeline: wan.WanT2V with config WAN_CONFIGS['t2v-A14B'] (generate.py:
  403-454); wan/configs/wan_t2v_A14B.py: two experts of the Wan2.1 14B DiT (dim 5120, 40 layers), boundary 0.875; sample_steps 40, sample_shift 12.0, sample_guide_scale (3.0, 4.0); frame_num 81 (shared_config.py);
  t5_cpu False, no FSDP, no sequence parallelism (single process);
- size 832*480 (SIZE_CONFIGS, wan/configs/__init__.py:21-30);
- solver 'unipc' (generate.py:202-206 default); seed fixed (the command
  leaves it random, :96; the seed does not change the compute);
- offload_model True, convert_model_dtype True (README.md:436).

Differences from the source, none of which changes the measured work: the
checkpoint is downloaded by its Hugging Face id instead of a local
--ckpt_dir; the video is not written to disk; no logging setup.
"""

KWARGS = {"size": "832*480", "frame_num": 81, "sampling_steps": 40,
          "shift": 12.0, "guide_scale": (3.0, 4.0), "seed": 42, "offload_model": True}


def generate_t2v_a14b(size: str = "832*480", frame_num: int = 81, sampling_steps: int = 40,
                      shift: float = 12.0, guide_scale=(3.0, 4.0), seed: int = 42,
                      offload_model: bool = True):
    # requires: pip install "git+https://github.com/Wan-Video/Wan2.2@1ea34ff"
    import time

    from huggingface_hub import snapshot_download
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

    ckpt_dir = snapshot_download("Wan-AI/Wan2.2-T2V-A14B")
    prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
              "fight intensely on a spotlighted stage.")

    t0 = time.time()
    pipe = wan.WanT2V(config=WAN_CONFIGS["t2v-A14B"], checkpoint_dir=ckpt_dir,
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_sp=False, t5_cpu=False, convert_model_dtype=True)
    t1 = time.time()
    video = pipe.generate(prompt, size=SIZE_CONFIGS[size], frame_num=frame_num,
                          shift=shift, sample_solver="unipc",
                          sampling_steps=sampling_steps, guide_scale=guide_scale,
                          seed=seed, offload_model=offload_model)
    t2 = time.time()
    return {"frames": int(video.shape[1]), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}


if __name__ == "__main__":
    print(generate_t2v_a14b(**KWARGS))
