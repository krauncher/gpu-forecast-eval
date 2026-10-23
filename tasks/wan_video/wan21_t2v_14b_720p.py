"""Wan2.1 text-to-video 14B, 1280x720, 81 frames, 50 steps, one GPU
(Wan-Video/Wan2.1, computational-efficiency table).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/wan_video/wan21_t2v_14b_720p.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
wan21_t2v_14b_720p.json; the form a service takes is in
variants/<service>/wan_video/wan21_t2v_14b_720p.py.

Source (the measured workload):
- github.com/Wan-Video/Wan2.1, commit 9737cba (2026-03-05), README.md
  "Computational Efficiency on Different GPUs" (lines 588-603) and
  assets/comp_effic.png, row T2V-14B / 720P, column "1" GPU.
- Command (README.md:171, with the table's settings at :598-601 — on 1 GPU
  `--offload_model True`, no prompt extension):
  python generate.py --task t2v-14B --size 1280*720
      --ckpt_dir ./Wan2.1-T2V-14B --offload_model True
      --prompt "Two anthropomorphic cats in comfy boxing gear and bright
      gloves fight intensely on a spotlighted stage."

This file calls the same pipeline the command runs; every value follows the
source (paths relative to the repository at 9737cba):
- pipeline: wan.WanT2V with config WAN_CONFIGS['t2v-14B'] (generate.py:
  360-369; wan/configs/wan_t2v_14B.py: DiT dim 5120, 40 layers, 40 heads;
  shared config bf16 parameters, umt5-xxl text encoder, text_len 512);
  t5_cpu False, no FSDP, no sequence parallelism (single process);
- size 1280x720 (SIZE_CONFIGS['1280*720'], wan/configs/__init__.py:32);
- frame_num 81 (generate.py:84-85 default for t2v);
- sampling_steps 50 (generate.py:71-72 default for t2v), shift 5.0
  (generate.py:76-77), solver 'unipc' (generate.py:229-231 default),
  guide_scale 5.0 (generate.py:242-244 default);
- negative prompt: the config's default (wan/text2video.py:168-169);
- offload_model True (README.md:599; generate.py:273-274 default on 1 GPU):
  the DiT is moved to the CPU after sampling and the text encoder after
  encoding (wan/text2video.py:178, 257-259, 265);
- seed fixed (the README command leaves it random, generate.py:91-92; the
  seed does not change the compute).

Differences from the source, none of which changes the measured work: the
checkpoint is downloaded by its Hugging Face id instead of a local
--ckpt_dir; the video is not written to disk (generate.py saves it after
generation); no logging setup.
"""

KWARGS = {"size": "1280*720", "frame_num": 81, "sampling_steps": 50,
          "shift": 5.0, "guide_scale": 5.0, "seed": 42, "offload_model": True}


def generate_t2v_14b(size: str = "1280*720", frame_num: int = 81, sampling_steps: int = 50,
                     shift: float = 5.0, guide_scale: float = 5.0, seed: int = 42,
                     offload_model: bool = True):
    # requires: pip install "git+https://github.com/Wan-Video/Wan2.1@9737cba"
    import time

    from huggingface_hub import snapshot_download
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS

    ckpt_dir = snapshot_download("Wan-AI/Wan2.1-T2V-14B")
    prompt = ("Two anthropomorphic cats in comfy boxing gear and bright gloves "
              "fight intensely on a spotlighted stage.")

    t0 = time.time()
    pipe = wan.WanT2V(config=WAN_CONFIGS["t2v-14B"], checkpoint_dir=ckpt_dir,
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_usp=False, t5_cpu=False)
    t1 = time.time()
    video = pipe.generate(prompt, size=SIZE_CONFIGS[size], frame_num=frame_num,
                          shift=shift, sample_solver="unipc",
                          sampling_steps=sampling_steps, guide_scale=guide_scale,
                          seed=seed, offload_model=offload_model)
    t2 = time.time()
    return {"frames": int(video.shape[1]), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}


if __name__ == "__main__":
    print(generate_t2v_14b(**KWARGS))
