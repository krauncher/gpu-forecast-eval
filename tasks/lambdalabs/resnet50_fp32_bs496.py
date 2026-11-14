"""ResNet-50 training, fp32, batch 496, synthetic data — Lambda GPU benchmark (v1).

Neutral form of the task: a plain script, runnable on any machine with a GPU
(`python tasks/lambdalabs/resnet50_fp32_bs496.py`). KWARGS are the call arguments. The
expected forecast and its sources are in resnet50_fp32_bs496.json; the form a service
takes is in variants/<service>/lambdalabs/resnet50_fp32_bs496.py.

Source (the measured workload):
- Benchmark: github.com/lambdal/deeplearning-benchmark, commit aa5addb6 (2025-12-17),
  pytorch/scripts/config_v1/config_pytorch_48GB.sh, PyTorch_resnet50_FP32_PARAMS: Classification/ConvNets main.py
  --arch resnet50 --epochs 1 --prof 50 --batch-size 496
  --training-only --data-backend synthetic --workers 64
  (RTX 6000 Ada, RTX A6000, 48 GB).
- Code it runs: github.com/LambdaLabsML/DeepLearningExamples, branch
  lambda/benchmark, PyTorch/Classification/ConvNets.

Reduction to one file, as in lambdalabs/resnet50_amp_bs1280.py:
- model: ResNet-50 v1.5 (layers [3, 4, 6, 3], stride on the 3x3 conv), the
  same topology as torchvision.models.resnet50;
- data: one synthetic batch, randn(batch, 3, 224, 224) on the GPU and random
  labels, yielded again every iteration (SynteticDataLoader); image size 224;
- iterations: --prof N runs N iterations per epoch, so epochs x prof =
  1 x 50 = 50 training steps; no evaluation (--training-only);
- step: forward and loss in fp32 (no --amp: no autocast, no loss scaler),
  loss.backward(), optimizer.step, zero_grad (training.py); PyTorch's
  default torch.backends.cudnn.allow_tf32 = True runs the convolutions in
  TF32 on Ampere and newer; loss.item() every step;
- optimizer and loss: SGD lr 0.1, momentum 0.9, weight decay 1e-4; CrossEntropyLoss;
  memory format NCHW (main.py default; no --memory-format in the config).

Differences from the source, none of which changes the step's compute:
torchvision's ResNet-50 instead of DLE's builder (same layers; weight init
differs); weight decay on BatchNorm parameters too; no 5 s sleep at the end of
each epoch; single process, no multiproc launcher (one GPU).
"""

KWARGS = {"batch_size": 496, "epochs": 1, "iterations_per_epoch": 50}


def train_resnet50_fp32(batch_size: int = 496, epochs: int = 1, iterations_per_epoch: int = 50):
    import time

    import torch
    import torch.nn as nn
    from torchvision.models import resnet50

    device = torch.device("cuda")
    model = resnet50(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (batch_size,), device=device)

    model.train()
    t0 = time.time()
    for epoch in range(epochs):
        for i in range(iterations_per_epoch):
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_value = loss.item()
        print(f"epoch {epoch + 1}/{epochs}: loss {loss_value:.3f}")
    elapsed = time.time() - t0
    steps = epochs * iterations_per_epoch
    return {"steps": steps, "elapsed_sec": round(elapsed, 2),
            "images_per_sec": round(steps * batch_size / elapsed, 1)}


if __name__ == "__main__":
    print(train_resnet50_fp32(**KWARGS))
