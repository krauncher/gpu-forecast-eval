"""Krauncher variant of tasks/timm/resnet152_amp_bs192.py (the neutral
form, with the source and the reduction it documents); the expected forecast
and its sources are in tasks/timm/resnet152_amp_bs192.json.

The task function is the neutral one, unchanged. FUNC is the task, OPTIONS
the arguments of its @client.task decorator, KWARGS the arguments of the
call. Synthetic data: nothing to download.
"""

OPTIONS = dict(
    timeout=1800,
)

KWARGS = dict(batch_size=192, steps=40)

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



FUNC = train_resnet152_amp
