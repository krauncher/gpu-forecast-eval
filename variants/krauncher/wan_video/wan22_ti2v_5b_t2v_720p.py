"""Krauncher variant of tasks/wan_video/wan22_ti2v_5b_t2v_720p.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/wan_video/wan22_ti2v_5b_t2v_720p.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=3600,
    data_urls=['hf://models/Wan-AI/Wan2.2-TI2V-5B'],
    pip=['git+https://github.com/Wan-Video/Wan2.2@1ea34ff'],
    stream_stderr=True,
)

KWARGS = dict(size="1280*704", frame_num=121, sampling_steps=50, shift=5.0,
              guide_scale=5.0, seed=42, offload_model=True)


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



FUNC = generate_ti2v_5b
