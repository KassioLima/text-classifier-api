# text-classifier-api

API FastAPI para classificação de textos.

Projeto para classificar automaticamente textos em 3 dimensões:

- `tipo`
- `produto`
- `assunto`

O objetivo é expor essa classificação via API para consumo do backend e reduzir esforço manual de triagem.

---

## 1) Contexto e necessidade

O cenário que motivou a ferramenta foi o tratamento de e-mails em texto livre. O corpo do e-mail precisa ser transformado em informação estruturada de forma rápida e consistente.

Este projeto resolve essa necessidade com uma API de classificação de textos que devolve, para cada dimensão (`tipo`, `produto`, `assunto`):

- `classId` (ID da classe)
- `label` (nome amigável da classe)
- `score` (confiança do modelo)

---

## 2) Como o projeto funciona

Fluxo em alto nível:

1. Recebe o texto de entrada.
2. Roda 3 modelos (um para cada dimensão).
3. Converte `LABEL_X` (Transformers) para `classId` via `classes.json`.
4. Traduz `classId` para `label` amigável via `shared/label_mappings.py`.
5. Retorna o resultado no endpoint `/classify`.

Arquivos principais:

- `api/main.py`
- `api/service.py`
- `api/models/base.py`
- `api/models/tipo_demanda.py`
- `api/models/produto.py`
- `api/models/assunto.py`

Organização da camada de modelos:

- `AiModelBase`: comportamento comum de carregamento de modelo/tokenizer/pipeline/classes.
- `AiModelTipoDemanda`, `AiModelProduto`, `AiModelAssunto`: herdam da base e definem configuração específica por dimensão.
- `AiModelService`: orquestra as inferências e monta a resposta da API.

## 2.1) Como funciona a seleção de modelos por classificação

Estado atual do projeto:

- Cada dimensão (`tipo`, `produto`, `assunto`) possui apenas **1 opção** em `modelOptions`.
- Essa opção atual é **modelo local em disco**:
- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/tipo`
- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/produto`
- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/assunto`

Implementação:

- `AiModelTipoDemanda.modelOptions`: `api/models/tipo_demanda.py`
- `AiModelProduto.modelOptions`: `api/models/produto.py`
- `AiModelAssunto.modelOptions`: `api/models/assunto.py`

Na inicialização da API (`api/main.py`), o código usa `modelIndex=0` para as 3 dimensões.
Como hoje existe só 1 opção por dimensão, `0` aponta para o modelo local.

## 2.2) Como usar modelo do Hugging Face (opcional)

Se quiser usar um modelo direto do Hugging Face Hub, adicione uma nova entrada em `modelOptions` da dimensão desejada.

Exemplo (segunda opção, índice `1`):

```python
{
  "path": "org-ou-usuario/nome-do-modelo-no-hub",
  "name": "nome-amigavel-no-retorno-da-api",
  "ACCESS_TOKEN": os.getenv("HUGGING_FACE_ACCESS_TOKEN_TIPO"),  # opcional (necessário se repo privado)
  "tokens": 512,
  "task": "sentiment-analysis",
  "labelMap": TIPO_DEMANDA_LABELS,
}
```

Depois, no startup da API, ajuste o índice:

- local: `AiModelTipoDemanda.init(modelIndex=0)`
- hub: `AiModelTipoDemanda.init(modelIndex=1)`

Observação importante:

- `ACCESS_TOKEN` em `modelOptions` é opcional.
- Para modelo local ou repositório público, ele pode ficar ausente/`None`.
- Para repositório privado no Hugging Face, ele deve ser preenchido com `os.getenv(...)`.

### Repositório privado no Hugging Face

Se o repositorio do modelo for privado, o `modelOptions` deve informar `ACCESS_TOKEN` carregado com `os.getenv(...)`.
Para modelos locais/públicos, essa chave pode não existir.

Exemplos:

- tipo: `"ACCESS_TOKEN": os.getenv("HUGGING_FACE_ACCESS_TOKEN_TIPO")`
- produto: `"ACCESS_TOKEN": os.getenv("HUGGING_FACE_ACCESS_TOKEN_PRODUTO")`
- assunto: `"ACCESS_TOKEN": os.getenv("HUGGING_FACE_ACCESS_TOKEN_ASSUNTO")`

Depois, crie um `.env` com base no `.env-example` e preencha os tokens:

- `HUGGING_FACE_ACCESS_TOKEN_TIPO=`
- `HUGGING_FACE_ACCESS_TOKEN_PRODUTO=`
- `HUGGING_FACE_ACCESS_TOKEN_ASSUNTO=`

Esses tokens são usados no carregamento (`from_pretrained`/`hf_hub_download`) para autenticar acesso a repositórios privados.

### 2.3) Quais modelos são esses (BERT, DeBERTa, etc.)

O nome `lr2e5_bs4_ep4_tipo30_produto30_assunto50` é o identificador do experimento treinado (hiperparâmetros + recortes mínimos por tarefa), e não o nome da arquitetura base.

Arquitetura do modelo promovido (em uso na API):

- Backbone: `neuralmind/bert-large-portuguese-cased` (**BERT Large em português**)
- Tipo de tarefa: `AutoModelForSequenceClassification` (3 cabeças separadas: `tipo`, `produto`, `assunto`)
- Script de treino do modelo promovido: `experiments/train_neuralmind_bert_large.py`

Outras arquiteturas testadas no projeto (não promovidas como modelo atual):

- `neuralmind/bert-base-portuguese-cased` (`experiments/train_neuralmind_bert.py`)
- `microsoft/mdeberta-v3-base` (`experiments/train_mdeberta.py`)
- `FacebookAI/xlm-roberta-large` (`experiments/train_xlm_roberta.py`)

Modelos atuais (locais):

- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/tipo`
- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/produto`
- `models/lr2e5_bs4_ep4_tipo30_produto30_assunto50/assunto`

---

## 3) Estrutura principal

- `api/`: API FastAPI de inferência.
- `api/models/`: classes de modelo (base comum + classes por dimensão).
- `datasets/`: dataset de entrada, splits, relatórios e scripts de preparo.
- `experiments/`: treino, avaliação e análise.
- `models/`: artefatos dos modelos treinados.
- `shared/`: mapeamentos de labels compartilhados.

---

## 4) Como executar (do dataset ao modelo)

### 4.1 Pré-requisitos

- Python 3.10+
- Ambiente virtual (`.venv`)

Instalação:

```powershell
python -m pip install -r requirements.txt
```

### 4.2 Preparar dataset de entrada

Coloque o dataset em:

- `datasets/dataset_train_ready.json`

Formato esperado (schema atual):

```json
[
  {
    "DetalhesDaDemanda": "texto para classificação",
    "TipoDeDemanda": 10,
    "produto": 6,
    "assunto": 1
  }
]
```

Observação: `DetalhesDaDemanda` e `TipoDeDemanda` são nomes legados do dataset.

### 4.3 Gerar datasets de treino e validação

```powershell
python datasets/scripts/init_datasets.py
```

Saidas principais:

- `datasets/dataset-tipo-train.csv`
- `datasets/dataset-tipo-evaluation.csv`
- `datasets/dataset-produto-train.csv`
- `datasets/dataset-produto-evaluation.csv`
- `datasets/dataset-assunto-train.csv`
- `datasets/dataset-assunto-evaluation.csv`

### 4.4 Treinar o modelo (configuração do modelo atual)

```powershell
python experiments/train_neuralmind_bert_large.py `
  --tasks tipo produto assunto `
  --epochs 4 `
  --batch-size 4 `
  --learning-rate 2e-5 `
  --max-length 512 `
  --min-train-samples-by-task tipo=30 produto=30 assunto=50 `
  --output-subdir lr2e5_bs4_ep4_tipo30_produto30_assunto50
```

### 4.5 Avaliar acerto simultâneo das 3 classificações (tipo + produto + assunto)

Esse passo mede o acerto conjunto por registro:

- `3/3`: acertou as 3 dimensões ao mesmo tempo.
- `2/3`, `1/3`, `0/3`: acertou parcialmente ou não acertou.

```powershell
python experiments/eval_joint_local_models.py `
  --dataset datasets/dataset_train_ready.json `
  --model-root models/lr2e5_bs4_ep4_tipo30_produto30_assunto50 `
  --output datasets/reports/joint_eval_full_lr2e5_bs4_ep4_tipo30_produto30_assunto50.json
```

### 4.6 Subir a API

```powershell
python api/startApi.py
```

Health check:

- `GET http://127.0.0.1:8000/apiStatus`

Classificacao:

- `POST http://127.0.0.1:8000/classify`

Payload:

```json
{
  "prompt": "texto para classificação"
}
```

---

## 5) O que foi testado ate chegar ao modelo atual

### 5.1 Famílias de modelo e scripts explorados

- `experiments/train_neuralmind_bert_large.py` (modelo promovido)
- `experiments/train_neuralmind_bert.py`
- `experiments/train_mdeberta.py`
- `experiments/train_xlm_roberta.py`

### 5.2 Estratégias de avaliação local

- Smoke test inicial (200 registros): `joint_eval_smoke_200.json`
- Avaliação full sem pos-regra: `joint_eval_full_without_postrule.json`
- Avaliação full com pos-regra de tipo: `joint_eval_full_with_tipo_postrule.json`
- Avaliação final do modelo atual: `joint_eval_full_lr2e5_bs4_ep4_tipo30_produto30_assunto50.json`
- Análise de confiança e calibração:
- `joint_eval_full_without_postrule_with_confidence.json`
- `calibration_summary.json`
- Graficos em `datasets/reports/`

### 5.3 Baseline com API externa (prompt engineering)

- `openai_gpt5_sample_100_eval.json`
- `openai_gpt5_sample_100_v2_eval.json`
- `openai_gpt5_sample_500_v2_eval.json`
- `openai_gpt5_sample_300_v3_seed123_eval.json`

---

## 6) Evolução de desempenho até o modelo atual

### 6.1 Evolução local (modelo em disco)

Fonte: relatórios `joint_eval*.json` em `datasets/reports/`.

| Etapa | Amostras | 3/3 | 2/3 | 1/3 | 0/3 | Tipo | Produto | Assunto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke inicial (`joint_eval_smoke_200.json`) | 200 | 47.50% | 41.00% | 10.50% | 1.00% | 77.00% | 81.50% | 76.50% |
| Full sem pos-regra (`joint_eval_full_without_postrule.json`) | 14.141 | 65.36% | 28.29% | 5.87% | 0.49% | 83.97% | 86.37% | 88.17% |
| Full com pos-regra tipo (`joint_eval_full_with_tipo_postrule.json`) | 14.141 | 62.80% | 30.02% | 6.60% | 0.58% | 80.50% | 86.37% | 88.17% |
| **Modelo atual** (`joint_eval_full_lr2e5_bs4_ep4_tipo30_produto30_assunto50.json`) | **14.141** | **65.36%** | **28.29%** | **5.87%** | **0.49%** | **83.97%** | **86.37%** | **88.17%** |

Observação: a pos-regra de tipo piorou o resultado conjunto, então a versão sem pos-regra foi mantida.

### 6.2 Evolução de prompt (baseline GPT-5)

Fonte: campo `accuracy` dos relatórios `openai_gpt5_sample*_eval.json`.

| Relatério | Amostras | 3/3 (`all`) | Tipo | Produto | Assunto |
|---|---:|---:|---:|---:|---:|
| `openai_gpt5_sample_100_eval.json` | 100 | 7.00% | 49.00% | 27.00% | 54.00% |
| `openai_gpt5_sample_100_v2_eval.json` | 100 | 11.00% | 54.00% | 49.00% | 57.00% |
| `openai_gpt5_sample_500_v2_eval.json` | 500 | 15.20% | 53.40% | 47.20% | 58.00% |
| `openai_gpt5_sample_300_v3_seed123_eval.json` | 300 | 16.33% | 53.67% | 53.67% | 57.33% |

---

## 7) Métricas do modelo atual

### 7.1 Qualidade (full eval local)

Fonte: `datasets/reports/joint_eval_full_lr2e5_bs4_ep4_tipo30_produto30_assunto50.json`

- Total avaliado: **14.141**
- Acurácia por dimensão:
- Tipo: **83.97%**
- Produto: **86.37%**
- Assunto: **88.17%**
- Acerto conjunto:
- **3/3:** **65.36%**
- **2/3:** **28.29%**
- **1/3:** **5.87%**
- **0/3:** **0.49%**

### 7.2 Métricas de treino (modelo promovido)

Fonte: `datasets/reports/training/lr2e5_bs4_ep4_tipo30_produto30_assunto50/training_summary.json`

- `tipo` (min samples 30): eval accuracy **73.42%**, eval F1 weighted **73.41%**
- `produto` (min samples 30): eval accuracy **77.39%**, eval F1 weighted **77.07%**
- `assunto` (min samples 50): eval accuracy **75.98%**, eval F1 weighted **75.74%**

### 7.3 Calibracao

Fonte: `datasets/reports/calibration_summary.json`

- ECE Tipo: **0.1350**
- ECE Produto: **0.0834**
- ECE Assunto: **0.0904**

### 7.4 Desempenho (tempo de resposta)

Medicoes observadas durante testes locais:

- Inferencia por dimensão (exemplo real):
- Tipo: ~32 ms
- Produto: ~40 ms
- Assunto: ~35 ms
- Soma inferência: ~107 ms
- Chamada fim-a-fim via backend C#: ~335 ms (incluindo overhead HTTP/app)
- Teste de volume (`500` requisições): ~50 s totais (cerca de **10 req/s**) em ambiente local de teste

---

## 8) Observacoes finais

- O projeto está organizado para operar com modelo local, sem dependência externa em produção.
- O mapeamento de labels está centralizado em `shared/label_mappings.py`.
- A camada de inferência foi separada em classes por dimensão com herança de uma base comum em `api/models/`, reduzindo duplicação e facilitando manutenção.
- Recomenda-se versionar novos ciclos de treino por `output-subdir` e sempre gerar relatório full comparável antes de promover modelo.
