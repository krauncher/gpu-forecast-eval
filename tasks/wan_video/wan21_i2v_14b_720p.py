"""Wan2.1 image-to-video 14B, 720P (max area 1280x720), 81 frames, 40 steps,
one GPU (Wan-Video/Wan2.1, computational-efficiency table).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/wan_video/wan21_i2v_14b_720p.py`); the model and the input image
by public ids / URLs. KWARGS are the call arguments. The expected forecast and
its sources are in wan21_i2v_14b_720p.json; the form a service takes is in
variants/<service>/wan_video/wan21_i2v_14b_720p.py.

Source (the measured workload):
- github.com/Wan-Video/Wan2.1, commit 9737cba (2026-03-05), README.md
  "Computational Efficiency on Different GPUs" (lines 588-603) and
  assets/comp_effic.png, row I2V-14B / 720P, column "1" GPU; note at :603
  (I2V samples 40 steps).
- Command (README.md:313, with the table's settings at :598-601 — on 1 GPU
  `--offload_model True`, no prompt extension):
  python generate.py --task i2v-14B --size 1280*720
      --ckpt_dir ./Wan2.1-I2V-14B-720P --image examples/i2v_input.JPG
      --offload_model True --prompt "<the example prompt>"

This file calls the same pipeline the command runs; every value follows the
source (paths relative to the repository at 9737cba):
- pipeline: wan.WanI2V with config WAN_CONFIGS['i2v-14B'] (generate.py:
  417-426); t5_cpu False, no FSDP, no sequence parallelism;
- max_area 1280*720 (MAX_AREA_CONFIGS['1280*720'], wan/configs/__init__.py:
  40); frame_num 81 (generate.py:84-85); sampling_steps 40 (generate.py:
  71-74, i2v default); shift 5.0 (generate.py:76-77: 3.0 only for 480p);
  solver 'unipc' (generate.py:229-231); guide_scale 5.0 (generate.py:242-244);
- prompt and image: EXAMPLE_PROMPT['i2v-14B'] (generate.py:35-40), the image
  examples/i2v_input.JPG of the repository;
- offload_model True (README.md:599; generate.py:273-274 default on 1 GPU);
- seed fixed (the command leaves it random; it does not change the compute).

Differences from the source, none of which changes the measured work: the
checkpoint by its Hugging Face id instead of --ckpt_dir, the image by its
raw URL at the commit; the video is not written to disk; no logging setup.
"""

KWARGS = {"size": "1280*720", "frame_num": 81, "sampling_steps": 40,
          "shift": 5.0, "guide_scale": 5.0, "seed": 42, "offload_model": True}


def generate_i2v_14b(size: str = "1280*720", frame_num: int = 81, sampling_steps: int = 40,
                     shift: float = 5.0, guide_scale: float = 5.0, seed: int = 42,
                     offload_model: bool = True):
    # requires: pip install "git+https://github.com/Wan-Video/Wan2.1@9737cba"
    import io
    import time
    import urllib.request

    from PIL import Image
    from huggingface_hub import snapshot_download
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS

    ckpt_dir = snapshot_download("Wan-AI/Wan2.1-I2V-14B-720P")
    url = ("https://raw.githubusercontent.com/Wan-Video/Wan2.1/9737cba/"
           "examples/i2v_input.JPG")
    img = Image.open(io.BytesIO(urllib.request.urlopen(url).read())).convert("RGB")
    prompt = ("Summer beach vacation style, a white cat wearing sunglasses sits on a "
              "surfboard. The fluffy-furred feline gazes directly at the camera with a "
              "relaxed expression. Blurred beach scenery forms the background featuring "
              "crystal-clear waters, distant green hills, and a blue sky dotted with white "
              "clouds. The cat assumes a naturally relaxed posture, as if savoring the sea "
              "breeze and warm sunlight. A close-up shot highlights the feline's intricate "
              "details and the refreshing atmosphere of the seaside.")

    t0 = time.time()
    pipe = wan.WanI2V(config=WAN_CONFIGS["i2v-14B"], checkpoint_dir=ckpt_dir,
                      device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False,
                      use_usp=False, t5_cpu=False)
    t1 = time.time()
    video = pipe.generate(prompt, img, max_area=MAX_AREA_CONFIGS[size],
                          frame_num=frame_num, shift=shift, sample_solver="unipc",
                          sampling_steps=sampling_steps, guide_scale=guide_scale,
                          seed=seed, offload_model=offload_model)
    t2 = time.time()
    return {"frames": int(video.shape[1]), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}


if __name__ == "__main__":
    print(generate_i2v_14b(**KWARGS))
