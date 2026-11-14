"""SDXL base 1.0 text-to-image, fp16 (fp16 variant, fp16-fix VAE), 1024x1024, 30 steps,
batch 1, eager (no torch.compile), one GPU (Runcrate, "SDXL images per dollar on four GPUs").

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/runcrate/sdxl_fp16_bs1.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
sdxl_fp16_bs1.json; the form a service takes is in
variants/<service>/runcrate/sdxl_fp16_bs1.py.

Source (the measured workload):
- dev.to/runcrate/sdxl-images-per-dollar-on-four-gpus-rtx-a6000-l40s-rtx-5090-
  and-h100-eager-vs-torchcompile-16no (posted 2026-10-27, edited 2026-11-04;
  originally runcrate.ai), table rows "eager, batch 1": RTX A6000 5.79, L40S
  3.48, RTX 5090 2.77, H100 PCIe 2.25 seconds per image.
- Setup as stated: stabilityai/stable-diffusion-xl-base-1.0, base only,
  torch_dtype=torch.float16, variant="fp16", VAE madebyollin/sdxl-vae-fp16-fix
  (fp16), 1024 x 1024, 30 steps, the pipeline's default scheduler and
  guidance, num_images_per_prompt=1, PyTorch SDPA attention, no xFormers /
  TensorRT; diffusers 0.42.1, torch 2.8.0+cu128, Python 3.12. Timing: one
  warm-up call, then 8 images, torch.cuda.synchronize() around the loop,
  time.time(); one run per GPU ("Treat differences under about 5% as noise").

This file calls the same pipeline with the same arguments. Differences from
the source, none of which changes the measured work: the image is not saved,
one timed call instead of a warm-up and eight.
"""

KWARGS = {"height": 1024, "width": 1024, "num_inference_steps": 30}


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


if __name__ == "__main__":
    print(generate_sdxl_fp16(**KWARGS))
