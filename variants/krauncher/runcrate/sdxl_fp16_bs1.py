"""Krauncher variant of tasks/runcrate/sdxl_fp16_bs1.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/runcrate/sdxl_fp16_bs1.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call.
"""

OPTIONS = dict(
    timeout=1800,
    data_urls=['hf://models/stabilityai/stable-diffusion-xl-base-1.0', 'hf://models/madebyollin/sdxl-vae-fp16-fix'],
    pip=['diffusers', 'transformers', 'accelerate'],
    stream_stderr=True,
)

KWARGS = dict(height=1024, width=1024, num_inference_steps=30)


def generate_sdxl_fp16(height: int = 1024, width: int = 1024, num_inference_steps: int = 30):
    import time

    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16,
        variant="fp16").to("cuda")
    prompt = "a photo of an astronaut riding a horse on a beach at sunset, detailed, 35mm"
    torch.cuda.synchronize()
    t0 = time.time()
    image = pipe(prompt, height=height, width=width,
                 num_inference_steps=num_inference_steps, num_images_per_prompt=1).images[0]
    torch.cuda.synchronize()
    return {"size": list(image.size), "generate_sec": round(time.time() - t0, 2)}


FUNC = generate_sdxl_fp16
