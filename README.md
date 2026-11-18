# PA3 — Poucos rótulos: fauna em câmeras trap com dados não rotulados

Repositório do **Programming Assignment 3** da disciplina de Aprendizado Profundo (FGV EMAp), utilizando o pacote derivado do **ENA24-detection** (21 espécies, orçamento padrão de 5 rótulos por classe = 105 imagens rotuladas e 6.505 imagens no pool de treino).

- **Backbone:** ResNet-18 (`11.187.285` parâmetros — abaixo do limite de 12M).
- **Trilha escolhida (Parte 1):** **Trilha A — Auto-supervisionado (SimCLR com perda NT-Xent)**.
- **Modelo final entregue:** `results/parte2/trilha_a_simclr_b5_s0.pt` (ResNet-18 ImageNet + pré-treino SimCLR `min_scale=0.08` + fine-tuning com `rotulos_5.csv`, selecionado pelo maior F1 macro no `val` entre as 3 seeds).

---

## 1. Ambiente

Requer Python 3.10+ (testado em Python 3.13). Para criar o ambiente virtual e instalar as dependências:

```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Os dados esperados estão em `pa3_dados/` (`images/train`, `images/val`, `images/test_public`, `labels/` e `classes.csv`). Caso esteja em outro caminho, defina a variável de ambiente `PA3_DATA`.

---

## 2. Um comando que treina

Para treinar o modelo da Trilha A (se `results/parte1/imagenet_ssl_backbone.pt` já existir, faz o fine-tuning nos 105 rótulos escolhendo a época pelo F1 macro no `val`; caso contrário, executa antes o pré-treino SimCLR):

```bash
python src/cli.py train --metodo trilha_a_simclr --budget 5 --seed 0
```

Os pesos treinados são salvos em `results/treinos/` (ignorado pelo git). O comando **não sobrescreve** o modelo entregue em `results/parte2/` e recusa um `.pt` já existente, a menos que se passe `--sobrescrever`.

Também é possível treinar os pisos da Parte 0 (`--metodo piso_resnet18` ou `--metodo piso_scratch`) e outros orçamentos (`--budget 1` ou `--budget 20`).

Para reproduzir **todos os experimentos, tabelas e figuras** das Partes 0, 1, 2 e 3 interativamente, abra e execute o notebook principal:

```bash
jupyter notebook notebooks/pa3.ipynb
```

---

## 3. Um comando que avalia

Para avaliar o modelo entregue no `test_public` (reportando F1 macro, acurácia, loss e número de parâmetros):

```bash
python src/cli.py eval --pesos results/parte2/trilha_a_simclr_b5_s0.pt --split test_public
```

---

## 4. Inferência em CPU (`predict.py`)

O script `predict.py` roda em CPU, sem acesso à internet, verifica e imprime o número de parâmetros do modelo (`11,187,285`) e gera o CSV com as colunas `image_id,label`:

```bash
python predict.py --imagens pa3_dados/images/test_public --saida preds.csv
```

---

## 5. Estrutura do Repositório

| Arquivo / Pasta | Descrição |
|---|---|
| `notebooks/pa3.ipynb` | Notebook único com todo o desenvolvimento, tabelas, figuras e análises das Partes 0, 1, 2 e 3 |
| `src/common.py` | Carregamento de dados (`CachedImages`), arquiteturas (`small_cnn`, `resnet18`), treino (`fit`) e métricas (`evaluate`) |
| `src/simclr.py` | Implementação do SimCLR (aumentações em batch, perda `nt_xent` e `pretrain` com checkpoint) |
| `src/cli.py` | CLI de linha de comando para treinar (`train`) e avaliar (`eval`) |
| `predict.py` | Script oficial de predição em CPU para avaliação no `test_private` |
| `curva_orcamento.csv` | Números consolidados da Parte 2 (`metodo,orcamento,seed,f1_macro,acuracia`) |
| `results/parte0/` | Pesos, históricos e figuras (curvas e matriz de confusão) do piso supervisionado |
| `results/parte1/` | Pesos do backbone SimCLR (`imagenet_ssl_backbone.pt`), logs de perda e tabelas do Linear Probe e aumentação |
| `results/parte2/` | Pesos e predições do modelo final (`trilha_a_simclr_b5_s0.pt`, `preds_trilha_a_simclr_b5_s0.csv`), `runs.csv`, curva de orçamento e ganho por classe. As predições das outras 5 rodadas do orçamento 5 rodaram no Colab e não estão versionadas (ver nota abaixo) |
| `results/parte3/` | Matriz de confusão do modelo final, pares mais confundidos, tabela de saliência e galerias de erros/acertos com Grad-CAM e oclusão |
| `AI_LOG.md` | Registro de uso de ferramentas de IA no trabalho |

**Nota de reprodutibilidade.** O ganho por classe (Parte 2.3) usa as predições no `test_public` das 6 rodadas do orçamento 5 (piso e Trilha A, 3 seeds). Só a da Trilha A com seed 0 (o modelo entregue) está no repositório; as outras 5 foram geradas no Colab e não foram recuperadas. Por isso, sem rodar de novo a seção 2.1 do notebook, a seção 2.3 carrega o `results/parte2/ganho_por_classe_b5.csv` calculado na execução original em vez de recalculá-lo.
