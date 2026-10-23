"""Wan2.1 text-to-video 1.3B, 832x480, 81 frames, 50 steps, one RTX 4090
with model offload and the text encoder on the CPU (Wan-Video/Wan2.1,
computational-efficiency table).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/wan_video/wan21_t2v_1_3b_480p.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
wan21_t2v_1_3b_480p.json; the form a service takes is in
variants/<service>/wan_video/wan21_t2v_1_3b_480p.py.

Source (the measured workload):
- github.com/Wan-Video/Wan2.1, commit 9737cba (2026-03-05), README.md
  "Computational Efficiency on Different GPUs" (lines 588-603) and
  assets/comp_effic.png, row 4090 / T2V-1.3B / 480P, column "1" GPU.
- Command (README.md:177, the single-4090 example; the table's settings at
  :600-601 — `--offload_model True --t5_cpu`, no prompt extension):
  python generate.py --task t2v-1.3B --size 832*480
      --ckpt_dir ./Wan2.1-T2V-1.3B --offload_model True --t5_cpu
      --sample_shift 8 --sample_guide_scale 6 --prompt "Two anthropomorphic
      cats in comfy boxing gear and bright gloves fight intensely on a
      spotlighted stage."

This file calls the same pipeline the command runs; every value follows the
source (paths relative to the repository at 9737cba):
- pipeline: wan.WanT2V with config WAN_CONFIGS['t2v-1.3B'] (generate.py:
  360-369; wan/configs/wan_t2v_1_3B.py); t5_cpu True (README.md:600);
- size 832x480 (SIZE_CONFIGS['832*480'], wan/configs/__init__.py:34);
  frame_num 81 (generate.py:84-85); sampling_steps 50 (generate.py:71-72);
  shift 8 and guide_scale 6 as in the command (README.md:177; neither
  changes the compute); solver 'unipc' (generate.py:229-231);
- offload_model True (README.md:600);
- seed fixed (the command leaves it random; it does not change the compute).

Differences from the source, none of which changes the measured work: the
checkpoint by its Hugging Face id instead of --ckpt_dir; the video is not
written to disk; no logging setup.
"""

KWARGS = {"size": "832*480", "frame_num": 81, "sampling_steps": 50,
          "shift": 8.0, "guide_scale": 6.0, "seed": 42, "offload_model": True,
          "t5_cpu": True}


def generate_t2v_1_3b(size: str = "832*480", frame_num: int = 81, sampling_steps: int = 50,
                      shift: float = 8.0, guide_scale: float = 6.0, seed: int = 42,
                      offload_model: bool = True, t5_cpu: bool = True):
    # requires: pip install "git+https://github.com/Wan-Video/Wan2.1@9737cba"
    import time

    from huggingface_hub import snapshot_download
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    ckpt_dir = snapshot_download("Wan-AI/Wan2.1-T2V-1.3B")
    prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
              "fight intensely on a spotlighted stage.")

    t0 = time.time()
    pipe = wan.WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=ckpt_dir,
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_usp=False, t5_cpu=t5_cpu)
    t1 = time.time()
    video = pipe.generate(prompt, size=SIZE_CONFIGS[size], frame_num=frame_num,
                          shift=shift, sample_solver="unipc",
                          sampling_steps=sampling_steps, guide_scale=guide_scale,
                          seed=seed, offload_model=offload_model)
    t2 = time.time()
    return {"frames": int(video.shape[1]), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}


if __name__ == "__main__":
    print(generate_t2v_1_3b(**KWARGS))
