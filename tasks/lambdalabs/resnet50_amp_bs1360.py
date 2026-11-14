"""ResNet-50 training, AMP, batch 1360, synthetic data — Lambda GPU benchmark (v2).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/resnet50_amp_bs1360.py`). KWARGS are the call arguments. The
expected forecast and its sources are in resnet50_amp_bs1360.json; the form a service
takes is in variants/<service>/lambdalabs/resnet50_amp_bs1360.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, branch master as fetched on 2026-11-12,
  pytorch/scripts/config_v2/config_pytorch_96GB.sh (via config_pytorch_1xGH200_96GB_v2.sh), PyTorch_resnet50_AMP_PARAMS: Classification/ConvNets main.py
  --arch resnet50 --amp --static-loss-scale 256 --epochs 2 --prof 100 --batch-size 1360
  --training-only --data-backend synthetic --workers 64
  (GH200 96 GB).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, PyTorch/Classification/ConvNets.

Reduction to one file, as in lambdalabs/resnet50_amp_bs1280.py:
- model: ResNet-50 v1.5 (layers [3, 4, 6, 3], stride on the 3x3 conv), the
  same topology as torchvision.models.resnet50;
- data: one synthetic batch, randn(batch, 3, 224, 224) on the GPU and random
  labels, yielded again every iteration (SynteticDataLoader); image size 224;
- iterations: --prof N runs N iterations per epoch, so epochs x prof =
  2 x 100 = 200 training steps; no evaluation (--training-only);
- step: forward and loss under torch.cuda.amp.autocast, scaler.scale(loss)
  .backward(), scaler.step, scaler.update, zero_grad (training.py:91-95,
  154-178); GradScaler(init_scale=256, growth_interval=1e9) = a static loss
  scale (main.py:482-488); loss.item() every step;
- optimizer and loss: SGD lr 0.1, momentum 0.9, weight decay 1e-4; CrossEntropyLoss;
  memory format NCHW (main.py default; no --memory-format in the config).

Differences from the source, none of which changes the step's compute:
torchvision's ResNet-50 instead of DLE's builder (same layers; weight init
differs); weight decay on BatchNorm parameters too; no 5 s sleep at the end of
each epoch; single process, no multiproc launcher (one GPU).
"""

KWARGS = {"batch_size": 1360, "epochs": 2, "iterations_per_epoch": 100}


def train_resnet50_amp(batch_size: int = 1360, epochs: int = 2, iterations_per_epoch: int = 100):
    import time

    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    device = torch.device("cuda")
    model = resnet50(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(init_scale=256, growth_interval=1_000_000_000)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        for i in range(iterations_per_epoch):
            with torch.cuda.amp.autocast():
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            loss_value = loss.item()
        print(f"epoch {epoch + 1}/{epochs}: loss {loss_value:.3f}")
    elapsed = time.time() - t0
    steps = epochs * iterations_per_epoch
    return {"steps": steps, "elapsed_sec": round(elapsed, 2),
            "images_per_sec": round(steps * batch_size / elapsed, 1)}


if __name__ == "__main__":
    print(train_resnet50_amp(**KWARGS))
