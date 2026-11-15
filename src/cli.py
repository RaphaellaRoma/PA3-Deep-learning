"""Interface de linha de comando para treinar e avaliar os modelos do PA3.

Exemplos de uso (conforme documentado no README.md):
  # Treinar o modelo final (Trilha A: SimCLR fine-tuning no orçamento 5)
  python src/cli.py train --metodo trilha_a_simclr --budget 5 --seed 0

  # Avaliar os pesos salvos no test_public
  python src/cli.py eval --pesos results/parte2/trilha_a_simclr_b5_s0.pt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common as C
import simclr as S


def build_model(metodo: str, nc: int, ssl_backbone_path: Path):
    if metodo == "piso_scratch":
        return C.small_cnn(nc)
    if metodo == "piso_resnet18":
        return C.resnet18(nc, pretrained=True)
    if metodo == "trilha_a_simclr":
        if not ssl_backbone_path.exists():
            print("Pré-treinando backbone SimCLR (imagenet_ssl)...")
            S.pretrain("imagenet", epochs=50, ckpt_dir=ssl_backbone_path.parent, name="imagenet_ssl", bs=256, img_size=160)
        m = C.resnet18(nc, pretrained=False)
        state = torch.load(ssl_backbone_path, map_location="cpu")
        m.load_state_dict(state, strict=False)
        return m
    raise ValueError(f"Método desconhecido: {metodo}")


def cmd_train(args):
    names, c2i = C.load_classes()
    nc = len(names)
    C.set_seed(args.seed)

    train_ds = C.CachedImages(C.read_split(f"rotulos_{args.budget}.csv", "train"), c2i, train=True)
    val_ds = C.CachedImages(C.read_split("val.csv", "val"), c2i, train=False)
    bs = min(args.bs, len(train_ds))
    train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(args.seed))
    val_dl = DataLoader(val_ds, 64)

    ssl_path = ROOT / "results" / "parte1" / "imagenet_ssl_backbone.pt"
    model = build_model(args.metodo, nc, ssl_path)
    lr = args.lr or (2e-3 if args.metodo == "piso_scratch" else 3e-4)
    epochs = args.epochs or (40 if args.metodo == "piso_scratch" else 20)

    print(f"Treinando {args.metodo} | orçamento={args.budget} | seed={args.seed} | params={C.count_params(model):,} | device={C.DEVICE}")
    hist, best_ep, best_state = C.fit(model, train_dl, val_dl, epochs=epochs, lr=lr, num_classes=nc)

    out_dir = ROOT / "results" / "parte2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = Path(args.saida) if args.saida else (out_dir / f"{args.metodo}_b{args.budget}_s{args.seed}.pt")
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_pt)
    pd.DataFrame(hist).to_csv(out_pt.with_name(f"hist_{out_pt.stem}.csv"), index=False)
    print(f"Melhor época (val F1): {best_ep} | Pesos salvos em: {out_pt}")


def cmd_eval(args):
    names, c2i = C.load_classes()
    nc = len(names)
    split_csv = f"{args.split}.csv"
    ds = C.CachedImages(C.read_split(split_csv, args.split), c2i, train=False)
    dl = DataLoader(ds, 64)

    weights_path = Path(args.pesos)
    model = C.small_cnn(nc) if "scratch" in weights_path.name else C.resnet18(nc, pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    res = C.evaluate(model, dl, "cpu", nc)
    print(f"Avaliação em {args.split} ({len(ds)} imagens) | Parâmetros: {C.count_params(model):,}")
    print(f"F1 macro: {res['f1_macro']:.4f} | Acurácia: {res['acc']:.4f} | Loss: {res['loss']:.4f}")


def main():
    p = argparse.ArgumentParser(description="Treino e avaliação do PA3.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="Treina um modelo (piso ou Trilha A) escolhendo a época pelo val")
    pt.add_argument("--metodo", choices=["piso_scratch", "piso_resnet18", "trilha_a_simclr"], default="trilha_a_simclr")
    pt.add_argument("--budget", type=int, choices=[1, 5, 20], default=5)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--epochs", type=int, default=None)
    pt.add_argument("--lr", type=float, default=None)
    pt.add_argument("--bs", type=int, default=32)
    pt.add_argument("--saida", default=None, help="Caminho opcional para salvar o .pt")

    pe = sub.add_parser("eval", help="Avalia um arquivo de pesos .pt em val ou test_public")
    pe.add_argument("--pesos", default=str(ROOT / "results" / "parte2" / "trilha_a_simclr_b5_s0.pt"))
    pe.add_argument("--split", choices=["val", "test_public"], default="test_public")

    args = p.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
