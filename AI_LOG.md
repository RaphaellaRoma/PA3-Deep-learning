# Registro de Uso de Inteligência Artificial (`AI_LOG.md`)

Em conformidade com a seção 5 do enunciado do PA3, descrevemos abaixo como ferramentas de IA (assistentes de código no VS Code / Antigravity) foram utilizadas ao longo do desenvolvimento do trabalho e como revisamos e validamos o que foi produzido.

---

## 1. Estruturação do Pipeline e Otimização para CPU/GPU (Parte 0 e Parte 1)

- **Episódio 1 — Gargalo de tempo da CNN do zero em CPU (Parte 0):**
  - *Problema:* Na primeira implementação de `small_cnn` (`src/common.py`), dois blocos de convolução $3\times 3$ operavam diretamente na resolução $224\times 224$ sem redução espacial inicial, levando ~28 minutos para 40 épocas em CPU.
  - *Uso da IA:* Recorremos ao assistente para diagnosticar o custo por camada e substituir o bloco de entrada por um *stem* $5\times 5$ com `stride=2` seguido de `MaxPool2d(2)` (reduzindo a resolução para $56\times 56$ logo na entrada, como na ResNet), mantendo ~1,17M de parâmetros e reduzindo o tempo em CPU para ~4,4 minutos (e <1 min em GPU). Guardamos o resultado anterior em `results/parte0/baseline_cnn_v1/` para comparação.

- **Episódio 2 — Pré-treino SimCLR e execução em kernel remoto do Colab (Parte 1):**
  - *Problema:* O pré-treino contrastivo (SimCLR com perda NT-Xent nas 6.505 imagens de `train`) era inviável na CPU local, e ao conectar o VS Code ao kernel remoto do Google Colab ocorreram dois problemas: (a) erro `No module named 'imp'` disparado pela extensão `%load_ext autoreload` no Python 3.13 do Colab; (b) as transformações `PIL` na CPU da máquina virtual atrasavam cada época para ~40s.
  - *Uso da IA:* Utilizamos a IA para isolar a falha do `autoreload`, migrar o pipeline de aumentação do SimCLR (`src/simclr.py`) para tensores `uint8` em RAM com `torchvision.transforms.v2` na GPU e implementar checkpoints retomáveis.

---

## 2. Revisão Crítica de Reprodutibilidade e Rigor nas Análises (Partes 1, 2 e 3)

- **Episódio 3 — Divergência entre saídas do notebook (GPU) e arquivos locais (CPU):**
  - *Problema:* Durante a revisão interna da dupla, identificamos que o notebook `notebooks/pa3.ipynb` exibia as saídas da execução em GPU no Colab, enquanto `results/parte0/` e `curva_orcamento.csv` ainda continham arquivos de uma rodada anterior em CPU, além de pesos `.pt` que haviam ficado retidos na VM remota ou bloqueados pelo `.gitignore`.
  - *Como resolvemos:* Sincronizamos todos os artefatos, históricos e figuras para refletirem a mesma execução, ajustamos o `.gitignore` para versionar os pesos essenciais (`results/parte1/imagenet_ssl_backbone.pt` e `results/parte2/trilha_a_simclr_b5_s0.pt`) e criamos os scripts de linha de comando (`predict.py` e `src/cli.py`).

- **Episódio 4 — Revisão crítica de textos interpretativos gerados com auxílio de IA:**
  - *Problema:* Uma versão preliminar dos textos de discussão no notebook afirmava que havia um "padrão claro" de ganho em espécies noturnas (infravermelho) e que o SimCLR "reduzia a variância pela metade", além de omitir a discussão direta de que o experimento de aumentação (`min_scale=0.40`) piorou o F1 no teste (`0,3625` vs `0,3711` no linear probe; `0,3918` vs `0,4182` no fine-tuning).
  - *Correção humana + IA:* Auditamos os números da tabela `ganho_por_classe_b5.csv` e `saliencia_quantitativa.csv` e reescrevemos as conclusões das seções 1.1, 1.2, 2.2, 2.3 e Parte 3 para refletir estritamente o que os dados suportam: assumindo o resultado negativo do pré-treino SimCLR sobre o ImageNet nos três orçamentos, apontando os contraexemplos de espécies 87%–100% noturnas que não tiveram ganho (`virginia_opossum`, `eastern_cottontail`, `red_fox`, `grey_fox`, correlação de Spearman fraca $r=0,249$) e alinhando a descrição visual do Grad-CAM com a métrica `massa_borda` (`0,31` a `0,35`).
