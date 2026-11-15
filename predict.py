"""Script de inferência em CPU para o PA3 (test_private / qualquer pasta de imagens .jpg).

Uso exigido pelo enunciado:
  python predict.py --imagens PASTA --saida preds.csv

- Escreve um CSV com colunas `image_id,label`, usando os nomes de `classes.csv`.
- Roda em CPU, sem internet, e imprime o número de parâmetros do modelo.
"""
import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from common import IMG_SIZE, MAX_PARAMS, MEAN, STD, count_params, load_classes, resnet18

DEFAULT_WEIGHTS = ROOT / "results" / "parte2" / "trilha_a_simclr_b5_s0.pt"
FALLBACK_WEIGHTS = ROOT / "results" / "parte0" / "resnet18_b5_s0.pt"


class FolderImages(Dataset):
    """Lê todos os arquivos .jpg de uma pasta em CPU, redimensionados para IMG_SIZE x IMG_SIZE."""

    def __init__(self, image_paths):
        self.paths = list(image_paths)
        self.tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return p.stem, self.tf(img)


def main():
    parser = argparse.ArgumentParser(description="Predição em CPU para imagens de câmeras trap (PA3).")
    parser.add_argument("--imagens", "-- –imagens", dest="imagens", required=True, help="Pasta com imagens .jpg")
    parser.add_argument("--saida", "-- –saida", dest="saida", required=True, help="Arquivo CSV de saída (image_id,label)")
    parser.add_argument("--pesos", default=None, help="Caminho opcional para os pesos .pt do modelo")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do lote em CPU")
    args = parser.parse_args()

    names, _ = load_classes()
    nc = len(names)

    weights_path = Path(args.pesos) if args.pesos else (DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else FALLBACK_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(f"Arquivo de pesos não encontrado: {weights_path}")

    model = resnet18(nc, pretrained=False)
    n_params = count_params(model)
    print(f"Modelo: ResNet-18 | Parâmetros: {n_params:,} (limite: {MAX_PARAMS:,}) | Pesos: {weights_path.relative_to(ROOT) if weights_path.is_relative_to(ROOT) else weights_path}")
    assert n_params <= MAX_PARAMS, f"Modelo excede o limite de {MAX_PARAMS} parâmetros!"

    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    model.to("cpu").eval()

    img_dir = Path(args.imagens)
    paths = sorted(img_dir.glob("*.jpg"))
    if not paths:
        paths = sorted(img_dir.glob("*.JPG"))

    ds = FolderImages(paths)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    rows = []
    with torch.no_grad():
        for stems, x in dl:
            preds = model(x).argmax(dim=1).tolist()
            for stem, idx in zip(stems, preds):
                rows.append((stem, names[idx]))

    out_path = Path(args.saida)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "label"])
        w.writerows(rows)

    print(f"Predições salvas em {out_path} ({len(rows)} imagens).")


if __name__ == "__main__":
    main()
