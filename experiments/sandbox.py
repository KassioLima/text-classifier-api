from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Sandbox de experimentação rápida com Hugging Face Trainer (foco em assunto).
dataset = load_dataset(
    "csv",
    data_files={
        "train": str(DATASETS_DIR / "dataset-assunto-train.csv"),
        "test": str(DATASETS_DIR / "dataset-assunto-evaluation.csv"),
    },
)

model_checkpoint = "KassioLima/autotrain-yrpuv-1x07g"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Criar dicionários para conversão das labels
labels = list(set(dataset["train"]["label"]))  # Obtém todas as categorias únicas
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

def tokenize_and_encode(examples):
    # Tokeniza texto e converte label textual para ID numérico.
    tokenized = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    tokenized["label"] = [label2id[label] for label in examples["label"]]
    return tokenized


# Aplicar a tokenização nos datasets de treino e teste
dataset = dataset.map(tokenize_and_encode, batched=True)
dataset = dataset.remove_columns(["text"])  # Remove a coluna original de texto
dataset.set_format("torch")  # Formato compatível com PyTorch

num_labels = len(labels)  # Número total de categorias
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir="./meu_modelo_treinado",
    evaluation_strategy="epoch",   # Avaliação no final de cada época
    save_strategy="epoch",         # Salvar o modelo a cada época
    per_device_train_batch_size=4, # Tamanho do batch no treino
    per_device_eval_batch_size=4,  # Tamanho do batch na avaliação
    num_train_epochs=5,            # Número de épocas
    weight_decay=0.01,             # Decaimento do peso (regularização)
    logging_dir="./logs",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],  # Conjunto de treino
    eval_dataset=dataset["test"],    # Conjunto de teste
    tokenizer=tokenizer
)

trainer.train()

# Exporta artefatos para reuso local após treino de prova de conceito.
model.save_pretrained("./meu_modelo_treinado")
tokenizer.save_pretrained("./meu_modelo_treinado")

metrics = trainer.evaluate()
print(metrics)
