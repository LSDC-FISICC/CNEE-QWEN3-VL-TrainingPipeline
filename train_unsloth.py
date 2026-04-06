import json
import torch
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset

print("=== Iniciando entrenamiento con Unsloth ===")

MAX_IMAGENES = 10  # Limitamos a 5 paginas por caso para caber en 8GB VRAM

# ── 1. Cargar modelo ──
model, tokenizer = FastVisionModel.from_pretrained(
    model_name="/home/julioefajardo/CNEE/models/Qwen3-VL-2B-Instruct",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
tokenizer.model_max_length = 8192

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    random_state=42,
)

print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9,2)} GB")

# ── 2. Cargar dataset ──
with open("/home/julioefajardo/CNEE/dataset/dataset_FINAL_100casos.json") as f:
    data = json.load(f)

BASE = "/home/julioefajardo/CNEE"

def construir_ejemplo(caso):
    imagenes = caso["images"][:MAX_IMAGENES]  # Limitar paginas
    rutas = [f"{BASE}/{img}" for img in imagenes]
    
    texto_user = ""
    for item in caso["messages"][0]["content"]:
        if item["type"] == "text" and item.get("text"):
            texto_user = item["text"]
    
    texto_assistant = ""
    for item in caso["messages"][1]["content"]:
        if item["type"] == "text" and item.get("text"):
            texto_assistant = item["text"]
    
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": ruta} for ruta in rutas] +
                       [{"type": "text",  "text": texto_user}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": texto_assistant}]
        }
    ]
    return {"messages": messages}

dataset_raw = [construir_ejemplo(c) for c in data["casos"]]
print(f"Casos preparados: {len(dataset_raw)}")
print(f"Max imagenes por caso: {MAX_IMAGENES}")

hf_dataset = Dataset.from_list(dataset_raw)
split = hf_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset  = split["test"]
print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

# ── 3. Entrenamiento ──
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=SFTConfig(
        output_dir="/home/julioefajardo/CNEE/output/qwen3vl_2b_cnee",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=3,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=8192,
    ),
)

print("Iniciando entrenamiento...")
trainer.train()

print("Guardando modelo...")
model.save_pretrained("/home/julioefajardo/CNEE/output/qwen3vl_2b_cnee/final")
tokenizer.save_pretrained("/home/julioefajardo/CNEE/output/qwen3vl_2b_cnee/final")
print("=== Entrenamiento completado ===")
