"""
Script de evaluacion del modelo Qwen3-VL fine-tuned
Ejecuta la evaluacion en el set de validacion
"""

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
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Evaluacion del Modelo Qwen3-VL ===\n")

# ══════════════════════════════════════════
# Configuracion
# ══════════════════════════════════════════
MAX_IMAGENES = 8
OUTPUT_DIR   = "/home/ubuntu/cnee/output/qwen3vl_4b_v2"
MODEL_PATH   = "/home/ubuntu/cnee/output/qwen3vl_4b_v2/final"  # Modelo fine-tuned con QLoRA
DATASET_PATH = "/home/ubuntu/cnee/CNEE-QWEN3-VL-TrainingPipeline/dataset/dataset_FINAL_rev_100casos.json"
BASE         = "/home/ubuntu/cnee/CNEE-QWEN3-VL-TrainingPipeline"


# ══════════════════════════════════════════
# 1. Cargar modelo
# ══════════════════════════════════════════
print("Cargando modelo...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
MODEL_MAX_SEQ_LENGTH = getattr(model, "max_seq_length", 4096)
try:
    tokenizer.model_max_length = MODEL_MAX_SEQ_LENGTH
except Exception:
    setattr(tokenizer, "model_max_length", MODEL_MAX_SEQ_LENGTH)

print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")

# ══════════════════════════════════════════
# 2. Cargar dataset y crear split
# ══════════════════════════════════════════
print("Cargando dataset...")
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


val_indices = []
for label, indices in grupos.items():
    n_val = max(1, int(len(indices) * 0.1))
    val_indices.extend(random.sample(indices, n_val))

# ══════════════════════════════════════════
# 3. Evaluacion del modelo en validacion
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
        truncation=False,
    ).to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
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
# 4. Grafica de evaluacion
# ══════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Evaluacion del Modelo Fine-tuned — Qwen3-VL 4B (CNEE)",
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

print("\n=== Evaluacion completada ===")
