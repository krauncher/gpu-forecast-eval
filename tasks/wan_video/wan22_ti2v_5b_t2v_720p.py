"""Wan2.2 text-image-to-video 5B in text-to-video mode, 1280x704, 121 frames, 50 steps,
one RTX 4090 with model offload, bf16 weights and the text encoder on the CPU (Wan-Video/Wan2.2, computational-efficiency table).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/wan_video/wan22_ti2v_5b_t2v_720p.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
wan22_ti2v_5b_t2v_720p.json; the form a service takes is in
variants/<service>/wan_video/wan22_ti2v_5b_t2v_720p.py.

Source (the measured workload):
- github.com/Wan-Video/Wan2.2, commit 1ea34ff (2026-09-21), README.md
  "Computational Efficiency on Different GPUs" (lines 426-440) and
  assets/comp_effic.png, row 4090 / TI2V-5B T2V / 720P, column "1" GPU.
- Command (README.md:229; the table's single-GPU settings at :436 —
  `--offload_model True --convert_model_dtype --t5_cpu`, no prompt extension):
  python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B
      --offload_model True --convert_model_dtype --t5_cpu --prompt "Two
      anthropomorphic cats in comfy boxing gear and bright gloves fight
      intensely on a spotlighted stage"

This file calls the same pipeline the command runs; every value follows the
source (paths relative to the repository at 1ea34ff):
- pipeline: wan.WanTI2V with config WAN_CONFIGS['ti2v-5B'] (generate.py:
  403-454); wan/configs/wan_ti2v_5B.py: DiT dim 3072, 30 layers, VAE stride (4, 16, 16); sample_steps 50, sample_shift 5.0, sample_guide_scale 5.0, frame_num 121;
  t5_cpu True, no FSDP, no sequence parallelism (single process);
- size 1280*704 (SIZE_CONFIGS, wan/configs/__init__.py:21-30);
- solver 'unipc' (generate.py:202-206 default); seed fixed (the command
  leaves it random, :96; the seed does not change the compute);
- offload_model True, convert_model_dtype True (README.md:436).

Differences from the source, none of which changes the measured work: the
checkpoint is downloaded by its Hugging Face id instead of a local
--ckpt_dir; the video is not written to disk; no logging setup.
"""

KWARGS = {"size": "1280*704", "frame_num": 121, "sampling_steps": 50,
          "shift": 5.0, "guide_scale": 5.0, "seed": 42, "offload_model": True}


def generate_ti2v_5b(size: str = "1280*704", frame_num: int = 121, sampling_steps: int = 50,
                     shift: float = 5.0, guide_scale=5.0, seed: int = 42,
                     offload_model: bool = True):
    # requires: pip install "git+https://github.com/Wan-Video/Wan2.2@1ea34ff"
    import time

    from huggingface_hub import snapshot_download
    import wan
    from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

    ckpt_dir = snapshot_download("Wan-AI/Wan2.2-TI2V-5B")
    prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
              "fight intensely on a spotlighted stage.")

    t0 = time.time()
    pipe = wan.WanTI2V(config=WAN_CONFIGS["ti2v-5B"], checkpoint_dir=ckpt_dir,
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_sp=False, t5_cpu=True, convert_model_dtype=True)
    t1 = time.time()
    video = pipe.generate(prompt, size=SIZE_CONFIGS[size], max_area=MAX_AREA_CONFIGS[size], frame_num=frame_num,
                          shift=shift, sample_solver="unipc",
                          sampling_steps=sampling_steps, guide_scale=guide_scale,
                          seed=seed, offload_model=offload_model)
    t2 = time.time()
    return {"frames": int(video.shape[1]), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}


if __name__ == "__main__":
    print(generate_ti2v_5b(**KWARGS))
