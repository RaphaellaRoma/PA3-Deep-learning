# Pacote de dados do PA3

Derivado do **ENA24-detection** (LILA BC), https://lila.science/datasets/ena24detection
Licença: Community Data License Agreement - Permissive 1.0,
https://cdla.dev/permissive-1-0/ . Citação pedida pelos autores:

> Yousif H, Kays R, He Z. Dynamic Programming Selection of Object Proposals for
> Sequence-Level Animal Species Classification in the Wild. IEEE TCSVT, 2019.

## Conteúdo

| Pasta / arquivo | O que é |
|---|---|
| `images/train/` | 6505 imagens. O pool. Sem rótulo, exceto as listadas em `labels/rotulos_K.csv` |
| `images/val/` | 202 imagens (até 10 por classe), com rótulo em `labels/val.csv` |
| `images/test_public/` | 811 imagens, com rótulo em `labels/test_public.csv` |
| `labels/rotulos_1.csv`, `rotulos_5.csv`, `rotulos_20.csv` | Orçamentos de 1, 5 e 20 rótulos por classe. Aninhados: todo id de `rotulos_1` está em `rotulos_5`, que está em `rotulos_20` |
| `classes.csv` | As 21 classes, com índice |

Todos os CSVs têm as colunas `image_id,label`. `image_id` é o nome do arquivo sem `.jpg`.
As imagens foram reduzidas para 512 px no lado maior, e a faixa de texto que as câmeras
imprimem (data, hora, nome da câmera) foi cortada. Por isso há imagens de proporções diferentes.

Existe um quarto conjunto, `test_private`, que fica com a monitoria e é usado para rodar o
`predict.py` de vocês.
