"""CogVideoX-2B text-to-video, 720x480, 49 frames, 50 steps, FP16, one GPU,
diffusers with the README's memory optimisations (THUDM/CogVideo).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/cogvideo/cogvideox_2b_t2v.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
cogvideox_2b_t2v.json; the form a service takes is in
variants/<service>/cogvideo/cogvideox_2b_t2v.py.

Source (the measured workload):
- github.com/THUDM/CogVideo, commit 7a1af71 (2025-11-04), README.md model
  table, row "Inference Speed (Step = 50, FP/BF16)", CogVideoX-2B column: 'Single A100: ~90 seconds', 'Single H100: ~45 seconds';
  tested with diffusers and all its memory optimisations (enable_sequential_cpu_offload, vae.enable_slicing, vae.enable_tiling; 'The inference speed tests also used the above memory optimization scheme').
- Script: inference/cli_demo.py at the same commit, generate_type "t2v",
  --model_path THUDM/CogVideoX-2b.

Every value follows the source (paths relative to the repository at 7a1af71):
- pipeline: diffusers CogVideoXPipeline.from_pretrained(model, torch_dtype=
  float16) (inference/cli_demo.py:121-122); --dtype float16: the README's recommended precision of CogVideoX-2B ('FP16*(Recommended)'); cli_demo's default is bfloat16 (:248-249);
  scheduler CogVideoXDPMScheduler with timestep_spacing "trailing" (:140-142);
- RESOLUTION_MAP['cogvideox-2b'] = (480, 720) (inference/cli_demo.py:41-49); frames 49 (CogVideoX 1.0, :80);
- num_inference_steps 50, guidance_scale 6.0, use_dynamic_cfg True, one video
  per prompt, seed 42 (:170-181, CLI defaults);
- memory optimisations as in the speed test: enable_sequential_cpu_offload,
  vae.enable_slicing, vae.enable_tiling (:150-152).

Differences from the source, none of which changes the measured work: the
prompt is a fixed example (cli_demo takes it from the command line); the video
is not exported to a file; no LoRA.
"""

KWARGS = {"num_frames": 49, "num_inference_steps": 50, "guidance_scale": 6.0,
          "height": 480, "width": 720, "seed": 42}


def generate_cogvideox_2b(num_frames: int = 49, num_inference_steps: int = 50,
                          guidance_scale: float = 6.0, height: int = 480, width: int = 720,
                          seed: int = 42):
    import time

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

    prompt = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
              "wooden stool in a serene bamboo forest, strumming a guitar.")

    t0 = time.time()
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
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


if __name__ == "__main__":
    print(generate_cogvideox_2b(**KWARGS))
