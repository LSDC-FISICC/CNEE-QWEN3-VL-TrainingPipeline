import json
import re
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Iniciando entrenamiento con Unsloth ===")

MAX_IMAGENES   = 5
OUTPUT_DIR     = "/home/julioefajardo/CNEE/output/qwen3vl_2b_cnee"
MODEL_PATH     = "/home/julioefajardo/CNEE/models/Qwen3-VL-2B-Instruct"
DATASET_PATH   = "/home/julioefajardo/CNEE/dataset/dataset_FINAL_100casos.json"
BASE           = "/home/julioefajardo/CNEE"

# ══════════════════════════════════════════
# 1. Cargar modelo
# ══════════════════════════════════════════
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
MODEL_MAX_SEQ_LENGTH = getattr(model, "max_seq_length", 2048)
try:
    tokenizer.model_max_length = MODEL_MAX_SEQ_LENGTH
except Exception:
    setattr(tokenizer, "model_max_length", MODEL_MAX_SEQ_LENGTH)

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

print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")

# ══════════════════════════════════════════
# 2. Cargar y preparar dataset
# ══════════════════════════════════════════
with open(DATASET_PATH) as f:
    data = json.load(f)

def construir_ejemplo(caso):
    imagenes = caso["images"][:MAX_IMAGENES]
    rutas    = [f"{BASE}/{img}" for img in imagenes]

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
    # Guardar label y texto_assistant para metricas
    label = caso["metadata"]["label"]
    return {"messages": messages, "label": label, "ground_truth": texto_assistant}

dataset_raw = [construir_ejemplo(c) for c in data["casos"]]
print(f"Casos preparados: {len(dataset_raw)}")

hf_dataset = Dataset.from_list(dataset_raw)

# Split 90/10 estratificado por label
from collections import defaultdict
import random
random.seed(42)

grupos = defaultdict(list)
for i, ej in enumerate(dataset_raw):
    grupos[ej["label"]].append(i)

val_indices = []
for label, indices in grupos.items():
    n_val = max(1, int(len(indices) * 0.1))
    val_indices.extend(random.sample(indices, n_val))

val_indices_set = set(val_indices)
train_indices   = [i for i in range(len(dataset_raw)) if i not in val_indices_set]

train_dataset = hf_dataset.select(train_indices).remove_columns(["label", "ground_truth"])
eval_dataset  = hf_dataset.select(val_indices).remove_columns(["label", "ground_truth"])
eval_raw      = [data["casos"][i] for i in val_indices]  # Casos originales para metricas

print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")
print(f"  Train APROBADOS:  {sum(1 for i in train_indices if dataset_raw[i]['label']=='APROBADO')}")
print(f"  Train RECHAZADOS: {sum(1 for i in train_indices if dataset_raw[i]['label']=='RECHAZADO')}")
print(f"  Val APROBADOS:    {sum(1 for i in val_indices   if dataset_raw[i]['label']=='APROBADO')}")
print(f"  Val RECHAZADOS:   {sum(1 for i in val_indices   if dataset_raw[i]['label']=='RECHAZADO')}")

# ══════════════════════════════════════════
# 3. Callback para registrar metricas
# ══════════════════════════════════════════
class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses  = []
        self.steps        = []
        self.eval_steps   = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_losses.append((step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_losses.append((step, logs["eval_loss"]))

metrics_callback = MetricsCallback()

class NoTruncUnslothVisionDataCollator(UnslothVisionDataCollator):
    def __init__(self, model, processor, *args, **kwargs):
        super().__init__(model, processor, *args, **kwargs)
        self.max_seq_length = None
        self.truncation = False

# ══════════════════════════════════════════
# 4. Entrenamiento
# ══════════════════════════════════════════
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=NoTruncUnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[metrics_callback],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=10,
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
        max_seq_length=MODEL_MAX_SEQ_LENGTH,
    ),
)

print("Iniciando entrenamiento...")
trainer.train()

# ══════════════════════════════════════════
# 5. Guardar modelo
# ══════════════════════════════════════════
print("Guardando modelo...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# ══════════════════════════════════════════
# 6. Calcular metricas sobre el set de validacion
# ══════════════════════════════════════════
print("\n=== Calculando metricas de validacion ===")
FastVisionModel.for_inference(model)

def extraer_decision(texto):
    """Extrae APROBADO o RECHAZADO del JSON generado."""
    match = re.search(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"', texto)
    if match:
        return match.group(1)
    if "APROBADO" in texto.upper():
        return "APROBADO"
    if "RECHAZADO" in texto.upper():
        return "RECHAZADO"
    return "DESCONOCIDO"

def inferir_caso(caso_raw):
    """Genera respuesta del modelo para un caso del set de validacion."""
    imagenes = caso_raw["images"][:MAX_IMAGENES]
    rutas    = [f"{BASE}/{img}" for img in imagenes]

    texto_user = ""
    for item in caso_raw["messages"][0]["content"]:
        if item["type"] == "text" and item.get("text"):
            texto_user = item["text"]

    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": ruta} for ruta in rutas] +
                       [{"type": "text",  "text": texto_user}]
        }
    ]

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    images = [Image.open(ruta).convert("RGB") for ruta in rutas]

    inputs = tokenizer(
        text=text_input,
        images=images,
        return_tensors="pt",
        truncation=False,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=False,
        )

    generated = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return generated

exact_matches    = 0
vqa_correct      = 0
resultados       = []

for i, caso_raw in enumerate(eval_raw):
    print(f"  Evaluando caso {i+1}/{len(eval_raw)}...", end="\r")

    ground_truth   = ""
    for item in caso_raw["messages"][1]["content"]:
        if item.get("type") == "text" and item.get("text"):
            ground_truth = item["text"]
    label_real     = caso_raw["metadata"]["label"]
    prediccion     = inferir_caso(caso_raw)
    label_pred     = extraer_decision(prediccion)

    # Exact match — respuesta identica (normalizada)
    exact = ground_truth.strip() == prediccion.strip()
    if exact:
        exact_matches += 1

    # VQA Accuracy — decision correcta (APROBADO/RECHAZADO)
    if label_pred == label_real:
        vqa_correct += 1

    resultados.append({
        "caso":        caso_raw.get("id", f"caso_{i}"),
        "label_real":  label_real,
        "label_pred":  label_pred,
        "exact_match": exact,
        "vqa_ok":      label_pred == label_real,
    })

n_val          = len(eval_raw)
exact_match_pct = exact_matches / n_val * 100
vqa_accuracy    = vqa_correct   / n_val * 100

print(f"\n{'='*50}")
print(f"METRICAS DE VALIDACION ({n_val} casos)")
print(f"{'='*50}")
print(f"Exact Match:    {exact_matches}/{n_val}  ({exact_match_pct:.1f}%)")
print(f"VQA Accuracy:   {vqa_correct}/{n_val}  ({vqa_accuracy:.1f}%)")
print(f"{'='*50}")

# Detalle por caso
print("\nDetalle por caso:")
print(f"{'Caso':<30} {'Real':<12} {'Pred':<12} {'VQA':>5} {'Exact':>7}")
print("-" * 70)
for r in resultados:
    vqa_str   = "OK" if r["vqa_ok"]      else "FAIL"
    exact_str = "OK" if r["exact_match"] else "FAIL"
    print(f"{r['caso']:<30} {r['label_real']:<12} {r['label_pred']:<12} {vqa_str:>5} {exact_str:>7}")

# Guardar metricas en JSON
metricas_output = {
    "n_val":          n_val,
    "exact_match":    exact_match_pct,
    "vqa_accuracy":   vqa_accuracy,
    "resultados":     resultados,
}
with open(f"{OUTPUT_DIR}/metricas_validacion.json", "w", encoding="utf-8") as f:
    json.dump(metricas_output, f, ensure_ascii=False, indent=2)
print(f"\nMetricas guardadas en: {OUTPUT_DIR}/metricas_validacion.json")

# ══════════════════════════════════════════
# 7. Grafica de Train / Val Loss + Metricas
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Resultados del Fine-tuning — Qwen3-VL 2B (CNEE)", fontsize=14, fontweight="bold")

# ── Plot 1: Train Loss ──
if metrics_callback.train_losses:
    steps_t, losses_t = zip(*metrics_callback.train_losses)
    axes[0].plot(steps_t, losses_t, color="#0D1B3E", linewidth=2, label="Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, "Sin datos de train loss", ha="center", va="center")
    axes[0].set_title("Training Loss")

# ── Plot 2: Val Loss ──
if metrics_callback.eval_losses:
    steps_e, losses_e = zip(*metrics_callback.eval_losses)
    axes[1].plot(steps_e, losses_e, color="#00B4D8", linewidth=2, marker="o", label="Val Loss")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, "Sin datos de val loss", ha="center", va="center")
    axes[1].set_title("Validation Loss")

# ── Plot 3: Metricas finales ──
metricas_names  = ["Exact Match", "VQA Accuracy"]
metricas_values = [exact_match_pct, vqa_accuracy]
colors          = ["#16A34A", "#00B4D8"]
bars = axes[2].bar(metricas_names, metricas_values, color=colors, width=0.5)
axes[2].set_title("Metricas de Validacion")
axes[2].set_ylabel("Porcentaje (%)")
axes[2].set_ylim(0, 110)
axes[2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, metricas_values):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{val:.1f}%",
        ha="center", va="bottom", fontweight="bold", fontsize=12
    )

plt.tight_layout()
plot_path = f"{OUTPUT_DIR}/resultados_entrenamiento.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafica guardada en: {plot_path}")
print("\n=== Entrenamiento y evaluacion completados ===")
