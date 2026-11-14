"""Krauncher variant of tasks/pytorch_blog/sdxl_bf16_a100.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/pytorch_blog/sdxl_bf16_a100.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=1800,
    data_urls=['hf://models/stabilityai/stable-diffusion-xl-base-1.0'],
    pip=['diffusers', 'transformers', 'accelerate'],
    stream_stderr=True,
)

KWARGS = dict(num_inference_steps=30)


def generate_sdxl_bf16(num_inference_steps: int = 30):
    import time

    import torch
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16).to("cuda")
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    torch.cuda.synchronize()
    t0 = time.time()
    image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
    torch.cuda.synchronize()
    return {"size": list(image.size), "generate_sec": round(time.time() - t0, 2)}


FUNC = generate_sdxl_bf16
