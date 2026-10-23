"""Krauncher variant of tasks/wan_video/wan21_i2v_14b_720p.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/wan_video/wan21_i2v_14b_720p.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=10800,
    data_urls=['hf://models/Wan-AI/Wan2.1-I2V-14B-720P'],
    pip=['git+https://github.com/Wan-Video/Wan2.1@9737cba'],
    stream_stderr=True,
)

KWARGS = dict(size="1280*720", frame_num=81, sampling_steps=40, shift=5.0, guide_scale=5.0, seed=42, offload_model=True)


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



FUNC = generate_i2v_14b
