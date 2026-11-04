"""SimCLR (Parte 1, Trilha A): pré-treino auto-supervisionado do backbone ResNet-18 no `train` (sem rótulos).

Pensado para rodar na GPU do Colab (retomável por checkpoint). Só os PESOS do backbone voltam para a CPU:
o fine-tuning, a avaliação e o predict.py ficam em CPU.
"""
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

from common import DATA, DEVICE, MEAN, STD, set_seed


def unlabeled_train_paths():
    """Todas as imagens de images/train. Os rótulos NÃO são lidos."""
    return sorted((DATA / "images" / "train").glob("*.jpg"))


class TwoViews(Dataset):
    """Cada item devolve duas views aumentadas da mesma imagem (sem rótulo).

    min_scale controla o menor recorte do RandomResizedCrop: é a alavanca do experimento de aumentação
    (com 0,08 o crop frequentemente não contém o animal; com valores maiores contém mais).
    """

    def __init__(self, paths, img_size=160, min_scale=0.08):
        self.images = [Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR) for p in paths]
        self.tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(), T.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.tf(self.images[i])


def nt_xent(z1, z2, temperature=0.5):
    """Perda NT-Xent (SimCLR). z1[i] e z2[i] são o par positivo; os demais 2N-2 são negativos."""
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def make_backbone(init: str = "imagenet") -> nn.Module:
    """ResNet-18 sem a camada final (saída: vetor de 512). init: 'imagenet' ou 'random'."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if init == "imagenet" else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Identity()
    return m


def pretrain(init, epochs, ckpt_dir, name, bs=256, lr=1e-3, temperature=0.5, img_size=160, min_scale=0.08,
             num_workers=2, seed=0, device=None, ckpt_every=2):
    """Pré-treina o backbone com SimCLR. Retomável: se existir checkpoint em ckpt_dir/name.ckpt, continua dele.

    Salva em ckpt_dir: <name>_backbone.pt (pesos do backbone, em CPU) e <name>_log.csv.
    """
    device = device or DEVICE
    set_seed(seed)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.ckpt"

    paths = unlabeled_train_paths()
    ds = TwoViews(paths, img_size, min_scale)
    dl = DataLoader(ds, bs, shuffle=True, drop_last=True, num_workers=num_workers,
                    persistent_workers=num_workers > 0, pin_memory=device == "cuda")
    backbone = make_backbone(init)
    head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
    model = nn.Sequential(backbone, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    log, start = [], 1
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        log, start = ck["log"], ck["epoch"] + 1
        print(f"retomando de {name}: época {start}/{epochs}")

    for ep in range(start, epochs + 1):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for v1, v2 in dl:
            v1, v2 = v1.to(device, non_blocking=True), v2.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                z = model(torch.cat([v1, v2]))
            loss = nt_xent(z[: len(v1)].float(), z[len(v1):].float(), temperature)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(v1)
            n += len(v1)
        sched.step()
        log.append({"epoch": ep, "loss": tot / n, "sec": time.perf_counter() - t0})
        print(f"[{name}] ep {ep}/{epochs} loss {tot / n:.4f} ({log[-1]['sec']:.0f}s)", flush=True)
        if ep % ckpt_every == 0 or ep == epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "log": log, "epoch": ep}, ckpt_path)

    cpu_state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    torch.save(cpu_state, ckpt_dir / f"{name}_backbone.pt")
    pd.DataFrame(log).to_csv(ckpt_dir / f"{name}_log.csv", index=False)
    return cpu_state, log

