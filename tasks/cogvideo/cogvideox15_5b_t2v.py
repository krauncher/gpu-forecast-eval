"""CogVideoX1.5-5B text-to-video, 1360x768, 81 frames (5 s at 16 fps), 50 steps, BF16,
one GPU, diffusers with the README's memory optimisations (THUDM/CogVideo).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/cogvideo/cogvideox15_5b_t2v.py`); the model by its public id.
KWARGS are the call arguments. The expected forecast and its sources are in
cogvideox15_5b_t2v.json; the form a service takes is in
variants/<service>/cogvideo/cogvideox15_5b_t2v.py.

Source (the measured workload):
- github.com/THUDM/CogVideo, commit 7a1af71 (2025-11-04), README.md model
  table, row "Inference Speed (Step = 50, FP/BF16)", CogVideoX1.5-5B column: 'Single A100: ~1000 seconds (5-second video)', 'Single H100: ~550 seconds (5-second video)';
  tested with diffusers and all its memory optimisations (enable_sequential_cpu_offload, vae.enable_slicing, vae.enable_tiling; 'The inference speed tests also used the above memory optimization scheme').
- Script: inference/cli_demo.py at the same commit, generate_type "t2v",
  --model_path zai-org/CogVideoX1.5-5B.

Every value follows the source (paths relative to the repository at 7a1af71):
- pipeline: diffusers CogVideoXPipeline.from_pretrained(model, torch_dtype=
  bfloat16) (inference/cli_demo.py:121-122); --dtype bfloat16 (cli_demo default, :248-249; README 'BF16 (Recommended)');
  scheduler CogVideoXDPMScheduler with timestep_spacing "trailing" (:140-142);
- RESOLUTION_MAP['cogvideox1.5-5b'] = (768, 1360) (inference/cli_demo.py:41-49); frames 81 = 5 s at 16 fps (:80, CLI default :229-230);
- num_inference_steps 50, guidance_scale 6.0, use_dynamic_cfg True, one video
  per prompt, seed 42 (:170-181, CLI defaults);
- memory optimisations as in the speed test: enable_sequential_cpu_offload,
  vae.enable_slicing, vae.enable_tiling (:150-152).

Differences from the source, none of which changes the measured work: the
prompt is a fixed example (cli_demo takes it from the command line); the video
is not exported to a file; no LoRA.
"""

KWARGS = {"num_frames": 81, "num_inference_steps": 50, "guidance_scale": 6.0,
          "height": 768, "width": 1360, "seed": 42}


def generate_cogvideox15_5b(num_frames: int = 81, num_inference_steps: int = 50,
                            guidance_scale: float = 6.0, height: int = 768, width: int = 1360,
                            seed: int = 42):
    import time

    import torch
    from diffusers import CogVideoXDPMScheduler, CogVideoXPipeline

    prompt = ("A panda, dressed in a small, red jacket and a tiny hat, sits on a "
              "wooden stool in a serene bamboo forest, strumming a guitar.")

    t0 = time.time()
    pipe = CogVideoXPipeline.from_pretrained("zai-org/CogVideoX1.5-5B", torch_dtype=torch.bfloat16)
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
    print(generate_cogvideox15_5b(**KWARGS))
