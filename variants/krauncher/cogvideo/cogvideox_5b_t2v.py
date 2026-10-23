"""Krauncher variant of tasks/cogvideo/cogvideox_5b_t2v.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/cogvideo/cogvideox_5b_t2v.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=3600,
    data_urls=['hf://models/THUDM/CogVideoX-5b'],
    pip=['diffusers', 'transformers', 'accelerate', 'sentencepiece'],
    stream_stderr=True,
)

KWARGS = dict(num_frames=49, num_inference_steps=50, guidance_scale=6.0,
              height=480, width=720, seed=42)


def generate_cogvideox_5b(num_frames: int = 49, num_inference_steps: int = 50,
                          guidance_scale: float = 6.0, height: int = 480, width: int = 720,
                          seed: int = 42):
    import time

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

    prompt = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
              "wooden stool in a serene bamboo forest, strumming a guitar.")

    t0 = time.time()
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config,
                                                       timestep_spacing="trailing")
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    t1 = time.time()
    frames = pipe(height=height, width=width, prompt=prompt, num_videos_per_prompt=1,
                  num_inference_steps=num_inference_steps, num_frames=num_frames,
                  use_dynamic_cfg=True, guidance_scale=guidance_scale,
                  generator=torch.Generator().manual_seed(seed)).frames[0]
    t2 = time.time()
    return {"frames": len(frames), "load_sec": round(t1 - t0, 1),
            "generate_sec": round(t2 - t1, 1)}



FUNC = generate_cogvideox_5b
