import json
import re
import os
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from transformers import TrainerCallback
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Entrenamiento con Train y Val Loss ===")

MAX_IMAGENES = 3
OUTPUT_DIR   = "/home/julioefajardo/CNEE/output/qwen3vl_2b_v2"
MODEL_PATH   = "/home/julioefajardo/CNEE/models/Qwen3-VL-2B-Instruct"
DATASET_PATH = "/home/julioefajardo/CNEE/dataset/dataset_FINAL_rev_100casos.json"
BASE         = "/home/julioefajardo/CNEE"
NUM_EPOCHS   = 25 # <-- Cambiar aqui

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 1. Cargar modelo
# ══════════════════════════════════════════
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
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
print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")

# ══════════════════════════════════════════
# 2. Cargar y dividir dataset
# ══════════════════════════════════════════
with open(DATASET_PATH) as f:
    data = json.load(f)

casos = data["casos"]

# Split estratificado 90/10 — misma semilla siempre
random.seed(42)
grupos = defaultdict(list)
for i, caso in enumerate(casos):
    grupos[caso["metadata"]["label"]].append(i)

val_indices = []
for label, indices in grupos.items():
    n_val = max(1, int(len(indices) * 0.1))
    val_indices.extend(random.sample(indices, n_val))

val_set   = set(val_indices)
train_indices = [i for i in range(len(casos)) if i not in val_set]

def construir_ejemplo(caso):
    imagenes = caso["images"][:MAX_IMAGENES]
    rutas    = [f"{BASE}/{img}" for img in imagenes]

    texto_user = ""
    for item in caso["messages"][0]["content"]:
        if item.get("type") == "text" and item.get("text"):
            texto_user = item["text"]

    texto_assistant = ""
    for item in caso["messages"][1]["content"]:
        if item.get("type") == "text" and item.get("text"):
            texto_assistant = item["text"]

    return {
        "messages": [
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
    }

train_dataset = Dataset.from_list([construir_ejemplo(casos[i]) for i in train_indices])
eval_dataset  = Dataset.from_list([construir_ejemplo(casos[i]) for i in val_indices])

print(f"Train: {len(train_dataset)} | Val: {len(eval_dataset)}")
print(f"  Train APROBADOS:  {sum(1 for i in train_indices if casos[i]['metadata']['label']=='APROBADO')}")
print(f"  Train RECHAZADOS: {sum(1 for i in train_indices if casos[i]['metadata']['label']=='RECHAZADO')}")
print(f"  Val APROBADOS:    {sum(1 for i in val_indices   if casos[i]['metadata']['label']=='APROBADO')}")
print(f"  Val RECHAZADOS:   {sum(1 for i in val_indices   if casos[i]['metadata']['label']=='RECHAZADO')}")

# ══════════════════════════════════════════
# 3. Callback para capturar train y val loss
# ══════════════════════════════════════════
class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_steps  = []
        self.train_losses = []
        self.val_steps    = []
        self.val_losses   = []
        self.epochs       = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.train_steps.append(step)
            self.train_losses.append(float(logs["loss"]))
            self.epochs.append(float(logs.get("epoch", 0)))
            print(f"  [Step {step:4d} | Epoch {logs.get('epoch', 0):.2f}] "
                  f"Train Loss: {logs['loss']:.4f}"
                  + (f"  |  Val Loss: {logs['eval_loss']:.4f}" if "eval_loss" in logs else ""))
        if "eval_loss" in logs:
            self.val_steps.append(step)
            self.val_losses.append(float(logs["eval_loss"]))

loss_callback = LossCallback()

# ══════════════════════════════════════════
# 4. Calcular eval_steps para evaluar
#    aproximadamente una vez por epoch
# ══════════════════════════════════════════
steps_per_epoch = max(1, len(train_dataset) // 16)  # batch_size efectivo = 16
eval_every      = max(1, steps_per_epoch)
total_steps     = steps_per_epoch * NUM_EPOCHS
print(f"\nSteps por epoch: ~{steps_per_epoch}")
print(f"Total steps:     ~{total_steps}")
print(f"Eval cada:       {eval_every} steps (1 vez por epoch)\n")

# ══════════════════════════════════════════
# 5. Entrenamiento
# ══════════════════════════════════════════
FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[loss_callback],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=1e-4,
        bf16=True,
        logging_steps=steps_per_epoch,   # Log una vez por epoch
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",           # Evaluar al final de cada epoch
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=8192,
    ),
)

# Liberar cache antes de entrenar
torch.cuda.empty_cache()
import gc
gc.collect()
print("Iniciando entrenamiento...")
result = trainer.train()

# ══════════════════════════════════════════
# 6. Guardar modelo
# ══════════════════════════════════════════
print("\nGuardando modelo...")
model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

# ══════════════════════════════════════════
# 7. Resumen de losses
# ══════════════════════════════════════════
print(f"\n{'='*55}")
print(f"RESUMEN DE LOSSES POR EPOCH")
print(f"{'='*55}")
print(f"{'Epoch':<8} {'Train Loss':<14} {'Val Loss':<14}")
print("-" * 40)

n = min(len(loss_callback.train_losses), len(loss_callback.val_losses))
for i in range(n):
    tl = loss_callback.train_losses[i]
    vl = loss_callback.val_losses[i]
    overfitting = " ⚠️  Posible overfitting" if vl > tl + 0.3 else ""
    print(f"{i+1:<8} {tl:<14.4f} {vl:<14.4f}{overfitting}")

# Si hay mas train logs que val logs
for i in range(n, len(loss_callback.train_losses)):
    print(f"{i+1:<8} {loss_callback.train_losses[i]:<14.4f} {'--':<14}")

# ══════════════════════════════════════════
# 8. Grafica de Train vs Val Loss + Metricas
# ══════════════════════════════════════════
tl_vals      = loss_callback.train_losses[:NUM_EPOCHS]
vl_vals      = loss_callback.val_losses[:NUM_EPOCHS]
epochs_range = list(range(1, NUM_EPOCHS + 1))

best_epoch   = int(vl_vals.index(min(vl_vals)) + 1) if vl_vals else None
best_val     = float(min(vl_vals))                   if vl_vals else 0.0
final_train  = float(tl_vals[-1])                    if tl_vals else 0.0
final_val    = float(vl_vals[-1])                    if vl_vals else 0.0
mejora_train = float(tl_vals[0] - tl_vals[-1])       if tl_vals else 0.0
mejora_val   = float(vl_vals[0] - vl_vals[-1])       if vl_vals else 0.0

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
fig.suptitle(f"Entrenamiento Qwen3-VL 2B — {NUM_EPOCHS} Epochs (CNEE)",
             fontsize=13, fontweight="bold")

# ── Plot 1: Train vs Val Loss por step ──
ax = axes[0]
if loss_callback.train_steps:
    ax.plot(loss_callback.train_steps, loss_callback.train_losses,
            color="#0D1B3E", linewidth=2, marker="o", markersize=5, label="Train Loss")
if loss_callback.val_steps:
    ax.plot(loss_callback.val_steps, loss_callback.val_losses,
            color="#DC2626", linewidth=2, marker="s", markersize=6,
            linestyle="--", label="Val Loss")
ax.set_title("Train vs Val Loss (por step)")
ax.set_xlabel("Steps")
ax.set_ylabel("Loss")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# ── Plot 2: Train vs Val Loss por epoch ──
ax2 = axes[1]
if tl_vals:
    ax2.plot(epochs_range[:len(tl_vals)], tl_vals,
             color="#0D1B3E", linewidth=2.5, marker="o", markersize=7, label="Train Loss")
if vl_vals:
    ax2.plot(epochs_range[:len(vl_vals)], vl_vals,
             color="#DC2626", linewidth=2.5, marker="s", markersize=7,
             linestyle="--", label="Val Loss")
if best_epoch:
    ax2.axvline(x=best_epoch, color="#16A34A", linestyle=":", linewidth=1.5,
                label=f"Mejor epoch ({best_epoch})")
    ax2.annotate(f"Val Loss\nmínimo\n{best_val:.4f}",
                 xy=(best_epoch, best_val),
                 xytext=(best_epoch + 0.3, best_val + 0.05),
                 fontsize=9, color="#16A34A",
                 arrowprops=dict(arrowstyle="->", color="#16A34A"))
ax2.set_title("Train vs Val Loss (por epoch)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.set_xticks(epochs_range)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# ── Plot 3: Metricas de entrenamiento ──
metricas_names  = [
    "Train Loss\nInicial",
    "Train Loss\nFinal",
    "Val Loss\nFinal",
    "Mejor\nVal Loss",
    "Mejora\nTrain",
    "Mejora\nVal",
]
metricas_values = [
    tl_vals[0]   if tl_vals else 0,
    final_train,
    final_val,
    best_val,
    mejora_train,
    mejora_val,
]
bar_colors = ["#6B7280", "#0D1B3E", "#DC2626", "#16A34A", "#00B4D8", "#F59E0B"]

bars = axes[2].bar(metricas_names, metricas_values, color=bar_colors, width=0.6)
axes[2].set_title("Resumen de Metricas de Entrenamiento")
axes[2].set_ylabel("Loss")
axes[2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, metricas_values):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.4f}",
        ha="center", va="bottom", fontweight="bold", fontsize=9
    )

plt.tight_layout()
plot_path = f"{OUTPUT_DIR}/train_val_loss.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nGrafica guardada en: {plot_path}")

# Guardar losses en JSON
losses_data = {
    "num_epochs":       NUM_EPOCHS,
    "train_steps":      loss_callback.train_steps,
    "train_losses":     loss_callback.train_losses,
    "val_steps":        loss_callback.val_steps,
    "val_losses":       loss_callback.val_losses,
    "best_epoch":       best_epoch,
    "best_val_loss":    best_val,
    "final_train_loss": final_train,
    "final_val_loss":   final_val,
    "mejora_train":     mejora_train,
    "mejora_val":       mejora_val,
}
with open(f"{OUTPUT_DIR}/losses.json", "w") as f:
    json.dump(losses_data, f, indent=2)
print(f"Losses guardados en: {OUTPUT_DIR}/losses.json")

# ══════════════════════════════════════════
# 9. Evaluacion del modelo en validacion
# ══════════════════════════════════════════
print("\n=== Evaluacion del modelo en set de validacion ===")

FastVisionModel.for_inference(model)

eval_casos = [casos[i] for i in val_indices]

def extraer_decision(texto):
    match = re.search(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"', texto)
    if match:
        return match.group(1)
    if "APROBADO" in texto.upper():
        return "APROBADO"
    if "RECHAZADO" in texto.upper():
        return "RECHAZADO"
    return "DESCONOCIDO"

def inferir_caso(caso):
    imagenes = caso["images"][:MAX_IMAGENES]
    rutas    = [f"{BASE}/{img}" for img in imagenes]

    texto_user = ""
    for item in caso["messages"][0]["content"]:
        if item.get("type") == "text" and item.get("text"):
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
        truncation=True,
        max_length=4096,
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

exact_matches = 0
vqa_correct   = 0
resultados    = []

for i, caso in enumerate(eval_casos):
    print(f"  Caso {i+1}/{len(eval_casos)}: {caso['id']}...")

    ground_truth = ""
    for item in caso["messages"][1]["content"]:
        if item.get("type") == "text" and item.get("text"):
            ground_truth = item["text"]

    label_real = caso["metadata"]["label"]
    prediccion = inferir_caso(caso)
    label_pred = extraer_decision(prediccion)

    exact  = ground_truth.strip() == prediccion.strip()
    vqa_ok = label_pred == label_real

    if exact:
        exact_matches += 1
    if vqa_ok:
        vqa_correct += 1

    print(f"    Real: {label_real} | Pred: {label_pred} | VQA: {'OK' if vqa_ok else 'FAIL'}")

    resultados.append({
        "caso":        caso["id"],
        "label_real":  label_real,
        "label_pred":  label_pred,
        "exact_match": exact,
        "vqa_ok":      vqa_ok,
        "prediccion":  prediccion,
    })

n_val           = len(eval_casos)
exact_match_pct = exact_matches / n_val * 100
vqa_accuracy    = vqa_correct   / n_val * 100

tp = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["label_pred"]=="APROBADO")
tn = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["label_pred"]=="RECHAZADO")
fp = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["label_pred"]=="APROBADO")
fn = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["label_pred"]=="RECHAZADO")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*55}")
print(f"METRICAS FINALES ({n_val} casos de validacion)")
print(f"{'='*55}")
print(f"VQA Accuracy:  {vqa_correct}/{n_val}  ({vqa_accuracy:.1f}%)")
print(f"Exact Match:   {exact_matches}/{n_val}  ({exact_match_pct:.1f}%)")
print(f"Precision:     {precision:.3f}")
print(f"Recall:        {recall:.3f}")
print(f"F1-Score:      {f1:.3f}")
print(f"{'='*55}")
print(f"\nMatriz de confusion:")
print(f"  TP (APROBADO correcto):  {tp}")
print(f"  TN (RECHAZADO correcto): {tn}")
print(f"  FP (falso APROBADO):     {fp}")
print(f"  FN (falso RECHAZADO):    {fn}")

metricas_output = {
    "n_val":          n_val,
    "vqa_accuracy":   vqa_accuracy,
    "exact_match":    exact_match_pct,
    "precision":      precision,
    "recall":         recall,
    "f1":             f1,
    "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "resultados":     resultados,
}
with open(f"{OUTPUT_DIR}/metricas_validacion.json", "w", encoding="utf-8") as f:
    json.dump(metricas_output, f, ensure_ascii=False, indent=2)
print(f"\nMetricas guardadas en: {OUTPUT_DIR}/metricas_validacion.json")

# ══════════════════════════════════════════
# 10. Grafica de evaluacion
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Evaluacion del Modelo Fine-tuned — Qwen3-VL 2B (CNEE)",
             fontsize=13, fontweight="bold")

# Plot 1: Metricas principales
metricas_names  = ["VQA\nAccuracy", "Exact\nMatch", "Precision", "Recall", "F1-Score"]
metricas_values = [vqa_accuracy, exact_match_pct, precision*100, recall*100, f1*100]
colors = ["#00B4D8", "#0D1B3E", "#16A34A", "#F59E0B", "#DC2626"]
bars = axes[0].bar(metricas_names, metricas_values, color=colors, width=0.6)
axes[0].set_title("Metricas de Evaluacion")
axes[0].set_ylabel("Porcentaje (%)")
axes[0].set_ylim(0, 120)
axes[0].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, metricas_values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f"{val:.1f}%",
        ha="center", va="bottom", fontweight="bold", fontsize=10
    )

# Plot 2: Matriz de confusion
confusion = [[tp, fn], [fp, tn]]
im = axes[1].imshow(confusion, cmap="Blues")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Pred APROBADO", "Pred RECHAZADO"])
axes[1].set_yticklabels(["Real APROBADO", "Real RECHAZADO"])
axes[1].set_title("Matriz de Confusion")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(confusion[i][j]),
                    ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if confusion[i][j] > n_val/4 else "black")

# Plot 3: Distribucion de predicciones
categorias = ["APROBADO\nCorrecto", "RECHAZADO\nCorrecto", "APROBADO\nIncorrecto", "RECHAZADO\nIncorrecto"]
valores    = [tp, tn, fp, fn]
colores    = ["#16A34A", "#0D1B3E", "#EF4444", "#F59E0B"]
bars2 = axes[2].bar(categorias, valores, color=colores, width=0.6)
axes[2].set_title("Distribucion de Predicciones")
axes[2].set_ylabel("Cantidad de casos")
axes[2].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars2, valores):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        str(val),
        ha="center", va="bottom", fontweight="bold", fontsize=12
    )

plt.tight_layout()
eval_plot_path = f"{OUTPUT_DIR}/evaluacion_modelo.png"
plt.savefig(eval_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafica guardada en: {eval_plot_path}")

print("\n=== Entrenamiento completado ===")
