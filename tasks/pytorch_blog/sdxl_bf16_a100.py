"""SDXL base 1.0 text-to-image, bfloat16, PyTorch SDPA attention, 30 steps, batch 1, one
A100 80GB (PyTorch blog "Accelerating Generative AI Part III: Diffusion, Fast").

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/pytorch_blog/sdxl_bf16_a100.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
sdxl_bf16_a100.json; the form a service takes is in
variants/<service>/pytorch_blog/sdxl_bf16_a100.py.

Source (the measured workload):
- pytorch.org/blog/accelerating-generative-ai-3 (page last updated 2024-11-14;
  code in github.com/huggingface/diffusion-fast), section "Use SDPA for
  performing attention computations": "SDPA gives a nice boost from 4.63
  seconds to 3.31 seconds". Setup: "a 400W 80GB A100 GPU (with its clock rate
  set to its maximum capacity)", torch 2.3.0.dev20231218+cu121, CUDA 12.1.
- Code of that step (verbatim in the post):
  pipe = StableDiffusionXLPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16).to("cuda")
  prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
  image = pipe(prompt, num_inference_steps=30).images[0]
  No attention processor is set: diffusers' default SDPA. Default resolution
  (SDXL: 1024x1024), guidance (5.0, CFG on) and scheduler.

Differences from the source, none of which changes the measured work: the
post's acknowledgements say the VAE madebyollin/sdxl-vae-fp16-fix "was used
throughout the benchmarking process", the code block loads the repository's
own VAE (the same architecture); this file follows the code block. The image
is not saved; the call is timed once (the post does not state its timing
method).
"""

KWARGS = {"num_inference_steps": 30}


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


if __name__ == "__main__":
    print(generate_sdxl_bf16(**KWARGS))
