"""SimCLR (Parte 1, Trilha A): pré-treino auto-supervisionado do backbone ResNet-18 no `train` (sem rótulos).

Otimizado para GPU: as imagens ficam em tensor uint8 na RAM e as aumentações (RandomResizedCrop, Flip,
ColorJitter, Grayscale) rodam em batch direto na GPU via torchvision.transforms.v2, reduzindo o tempo por época
de ~40s para ~3-5s no Colab. Se os pesos finais já existirem em disco e force=False, apenas carrega e imprime.
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
from torchvision.transforms import v2

from common import DATA, DEVICE, MEAN, STD, set_seed


def unlabeled_train_paths():
    """Todas as imagens de images/train. Os rótulos NÃO são lidos."""
    return sorted((DATA / "images" / "train").glob("*.jpg"))


class FastTensorImages(Dataset):
    """Guarda todas as imagens redimensionadas como tensores uint8 (C, H, W) na RAM para leitura instantânea."""

    def __init__(self, paths, img_size=160):
        to_u8 = v2.PILToTensor()
        self.tensors = [
            to_u8(Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR))
            for p in paths
        ]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i]


def make_gpu_aug(img_size=160, min_scale=0.08, device="cpu"):
    """Pipeline de aumentação SimCLR que roda direto em batch (ou tensor) na GPU."""
    return v2.Compose([
        v2.RandomResizedCrop(img_size, scale=(min_scale, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]).to(device)


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


_IMAGE_CACHE = {}


def get_cached_dataset(img_size=160):
    if img_size not in _IMAGE_CACHE:
        paths = unlabeled_train_paths()
        print(f"Carregando {len(paths)} imagens ({img_size}x{img_size}) na RAM...", flush=True)
        _IMAGE_CACHE[img_size] = FastTensorImages(paths, img_size)
    return _IMAGE_CACHE[img_size]


def pretrain(init, epochs, ckpt_dir, name, bs=256, lr=1e-3, temperature=0.5, img_size=160, min_scale=0.08,
             num_workers=0, seed=0, device=None, ckpt_every=5, force=False):
    """Pré-treina o backbone com SimCLR.

    - Se <name>_backbone.pt e <name>_log.csv já existirem e force=False, carrega direto sem treinar de novo.
    - Caso contrário, treina (retomando de <name>.ckpt se existir) e salva os pesos em CPU.
    """
    device = device or DEVICE
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_pt = ckpt_dir / f"{name}_backbone.pt"
    log_csv = ckpt_dir / f"{name}_log.csv"
    ckpt_path = ckpt_dir / f"{name}.ckpt"

    if not force and final_pt.exists() and log_csv.exists():
        cpu_state = torch.load(final_pt, map_location="cpu")
        log = pd.read_csv(log_csv).to_dict("records")
        print(f"[{name}] Resultados prontos carregados de {final_pt.name} ({len(log)} épocas, loss final={log[-1]['loss']:.4f}).")
        return cpu_state, log

    set_seed(seed)
    ds = get_cached_dataset(img_size)
    dl = DataLoader(ds, bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=(device == "cuda"))
    aug = make_gpu_aug(img_size, min_scale, device=device)

    backbone = make_backbone(init)
    head = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Linear(512, 128))
    model = nn.Sequential(backbone, head).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    log, start = [], 1
    if not force and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        log, start = ck["log"], ck["epoch"] + 1
        print(f"Retomando {name} da época {start}/{epochs}")

    for ep in range(start, epochs + 1):
        model.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x_u8 in dl:
            x_u8 = x_u8.to(device, non_blocking=True)
            # Gera as duas views na GPU (imagem a imagem para sortear crops independentes, ou em sub-blocos)
            v1 = torch.stack([aug(img) for img in x_u8])
            v2 = torch.stack([aug(img) for img in x_u8])
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
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
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"[{name}] ep {ep:02d}/{epochs} loss {tot / n:.4f} ({log[-1]['sec']:.1f}s)", flush=True)
        if ep % ckpt_every == 0 or ep == epochs:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "log": log, "epoch": ep}, ckpt_path)

    cpu_state = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    torch.save(cpu_state, final_pt)
    pd.DataFrame(log).to_csv(log_csv, index=False)
    return cpu_state, log
