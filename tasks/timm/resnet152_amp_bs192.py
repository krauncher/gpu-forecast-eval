"""ResNet-152 training step benchmark, AMP, batch 192, synthetic input — timm benchmark.py, RTX 3090.

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/timm/resnet152_amp_bs192.py`). KWARGS are the call arguments.
The expected forecast and its sources are in resnet152_amp_bs192.json; the
form a service takes is in variants/<service>/timm/resnet152_amp_bs192.py.

Source (the measured workload):
- Result: github.com/huggingface/pytorch-image-models, results/
  benchmark-train-amp-nchw-pt112-cu113-rtx3090.csv (PyTorch 1.12, CUDA 11.3,
  RTX 3090, AMP, NCHW, by the file name), row resnet152: 370.08 train samples/s,
  train_step_time 516.516 ms, batch 192, image 224.
- Code it runs: benchmark.py (tag v0.6.7), TrainBenchmarkRunner: model.train();
  per step zero_grad, then forward and CrossEntropyLoss(output, random targets)
  .backward() under torch.cuda.amp.autocast (no GradScaler), then
  optimizer.step(); the optimizer is create_optimizer_v2(model, opt='sgd',
  lr=1e-4); the input randn(batch, 3, 224, 224) is made once and reused;
  --num-warm-iter 10 untimed steps, --num-bench-iter 40 timed steps; the batch
  starts at 256 and decays on OOM (the row ran with 192).

Reduction to one file: torchvision's ResNet-152 (the same topology as timm's
resnet152); SGD lr 1e-4 with momentum 0.9 (timm's SGD momentum default, not
verified; the optimizer step is a small share of the step); the 40 timed steps
only (the 10 warm-up steps are outside the measured throughput).
"""

KWARGS = {"batch_size": 192, "steps": 40}


def train_resnet152_amp(batch_size: int = 192, steps: int = 40):
    import time

    import torch
    import torch.nn as nn
    from torchvision.models import resnet152

    device = torch.device("cuda")
    model = resnet152(num_classes=1000).to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    images = torch.randn(batch_size, 3, 224, 224, device=device)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(images)
            target = torch.empty(batch_size, dtype=torch.long, device=device).random_(1000)
            criterion(output, target).backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"steps": steps, "elapsed_sec": round(elapsed, 2),
            "images_per_sec": round(steps * batch_size / elapsed, 1)}


if __name__ == "__main__":
    print(train_resnet152_amp(**KWARGS))
