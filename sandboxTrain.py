# Importações necessárias
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer, BitsAndBytesConfig,
)
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import numpy as np
import joblib

# 1. Carregar os datasets
train_df = pd.read_csv("./datasets/dataset-assunto-train.csv")
test_df = pd.read_csv("./datasets/dataset-assunto-evaluation.csv")

# 2. Pré-processamento dos dados
# Codificar as labels para valores numéricos
label_encoder = LabelEncoder()
train_df["label"] = label_encoder.fit_transform(train_df["label"])
test_df["label"] = label_encoder.transform(test_df["label"])

# Converter os DataFrames para o formato de Dataset do Hugging Face
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Carregar o tokenizer e o modelo
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Definir o token de padding

# Função para tokenizar os textos
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,  # Ajuste o comprimento máximo conforme necessário
        return_tensors="pt",
    )

# Aplicar a tokenização aos datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 4. Carregar o modelo com head de classificação
num_labels = len(label_encoder.classes_)  # Número de classes
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="single_label_classification",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,  # Ou load_in_4bit=True para mais economia
    )
)

model.gradient_checkpointing_enable()

# 5. Configurar os argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",  # Diretório para salvar os resultados
    num_train_epochs=3,  # Número de épocas
    per_device_train_batch_size=1,  # Tamanho do batch de treino
    per_device_eval_batch_size=1,  # Tamanho do batch de avaliação
    learning_rate=2e-5,  # Taxa de aprendizado
    weight_decay=0.01,  # Decaimento do peso
    evaluation_strategy="epoch",  # Avaliar a cada época
    logging_dir="./logs",  # Diretório para logs
    save_strategy="epoch",  # Salvar o modelo a cada época
    fp16=True,  # Ativar precisão mista (se suportado pela GPU)
    gradient_accumulation_steps=8,  # Acumular gradientes para economizar memória
    logging_steps=10,  # Log a cada 10 passos
    save_total_limit=2,  # Limitar o número de checkpoints salvos
    report_to="none",  # Desativar relatórios externos (ou use "wandb" ou "tensorboard")
)

# 6. Função para calcular métricas de avaliação
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# 7. Criar o Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Salvar o LabelEncoder
joblib.dump(label_encoder, "label_encoder.pkl")

# 8. Treinar o modelo
trainer.train()

# 9. Salvar o modelo e o tokenizer
model.save_pretrained("./modelo_salvo")
tokenizer.save_pretrained("./modelo_salvo")

# 10. Exemplo de inferência após o treinamento
from transformers import pipeline

label_encoder = joblib.load("label_encoder.pkl")

# Carregar o pipeline de classificação de texto
classifier = pipeline(
    "text-classification",
    model="./modelo_salvo",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Usar GPU se disponível
)

# Testar com um exemplo
sample_text = "Seu texto de exemplo aqui"
prediction = classifier(sample_text)
predicted_label = label_encoder.inverse_transform([int(prediction[0]["label"].split("_")[-1])])
print(f"Texto: {sample_text} | Classe prevista: {predicted_label[0]}")