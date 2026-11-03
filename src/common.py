"""Utilidades compartilhadas: dados, modelos e métricas do PA3."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from torchvision import models, transforms as T

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "pa3_dados"
IMG_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
MAX_PARAMS = 12_000_000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_classes():
    df = pd.read_csv(DATA / "classes.csv")
    names = df.sort_values("idx")["label"].tolist()
    return names, {n: i for i, n in enumerate(names)}


def read_split(csv_name: str, image_dir: str):
    df = pd.read_csv(DATA / "labels" / csv_name)
    return [(DATA / "images" / image_dir / f"{i}.jpg", l) for i, l in zip(df.image_id, df.label)]


class CachedImages(Dataset):
    """Carrega as imagens uma vez, já em img_size x img_size (resize simples)."""

    def __init__(self, items, class_to_idx, train: bool, img_size: int = IMG_SIZE):
        self.images = [Image.open(p).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                       for p, _ in items]
        self.labels = [class_to_idx[l] for _, l in items]
        norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
        if train:
            self.tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.3, 0.3, 0.2),
                *norm,
            ])
        else:
            self.tf = T.Compose(norm)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.MaxPool2d(2))


def small_cnn_v1(num_classes: int) -> nn.Module:
    """CNN original (v1): convoluções a 224x224 sem redução inicial. Lenta em CPU; mantida só p/ comparação."""
    return nn.Sequential(_block(3, 32), _block(32, 64), _block(64, 128), _block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))


def small_cnn(num_classes: int) -> nn.Module:
    """CNN pequena treinada do zero (versão a do piso).

    Stem 5x5 com stride 2 + max-pool reduz 224 -> 56 logo na entrada (como a ResNet),
    o que corta o custo por imagem em CPU. O restante é igual à v1.
    """
    stem = nn.Sequential(nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(32),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))
    return nn.Sequential(stem, _block(32, 64), _block(64, 128), _block(128, 256),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))


def resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    ys, ps, loss_sum = [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y in loader:
        out = model(x.to(device))
        loss_sum += ce(out, y.to(device)).item()
        ps += out.argmax(1).cpu().tolist()
        ys += y.tolist()
    return {
        "loss": loss_sum / len(ys),
        "acc": accuracy_score(ys, ps),
        "f1_macro": f1_score(ys, ps, average="macro", labels=list(range(num_classes)), zero_division=0),
        "y": ys, "pred": ps,
    }



def fit(model, train_dl, val_dl, epochs, lr, num_classes, weight_decay=1e-2, verbose=True):
    """Treina em CPU e guarda o estado da melhor época segundo o F1 macro no val.

    Retorna (history, best_epoch, best_state). Não toca no test_public.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    ce = nn.CrossEntropyLoss()
    history, best = [], {"f1": -1}
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in train_dl:
            out = model(x)
            loss = ce(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        v = evaluate(model, val_dl, "cpu", num_classes)
        history.append({"epoch": ep, "train_loss": tl / n, "train_acc": tc / n,
                        "val_loss": v["loss"], "val_acc": v["acc"], "val_f1": v["f1_macro"]})
        if verbose:
            h = history[-1]
            print(f"ep {ep:02d} | treino loss {h['train_loss']:.3f} acc {h['train_acc']:.3f} | "
                  f"val loss {h['val_loss']:.3f} acc {h['val_acc']:.3f} F1 {h['val_f1']:.3f}", flush=True)
        if v["f1_macro"] > best["f1"]:
            best = {"f1": v["f1_macro"], "epoch": ep, "state": {k: t.clone() for k, t in model.state_dict().items()}}
    return history, best["epoch"], best["state"]
