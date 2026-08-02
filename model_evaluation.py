"""
Script de evaluacion del modelo Qwen3-VL fine-tuned (v3) con escalacion
por reglas y confidence recalculado.

FLUJO:
  1. Inferencia del modelo → texto JSON predicho
  2. Re-contar criterios cumplidos (cumple:true en criterios.{})
  3. Aplicar reglas del prompt CNEE para decision_regla
  4. Calcular confidence_real basado en criterios
  5. Si confidence_real < 0.75 → REQUIERE_REVISION_MANUAL
  6. Reportar metricas: modelo original Y sistema con escalacion
"""

import json
import re
import os
import random
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from unsloth import FastVisionModel
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Evaluacion del Modelo Qwen3-VL con Escalacion por Reglas ===\n")

# ══════════════════════════════════════════
# Configuracion
# ══════════════════════════════════════════
MAX_IMAGENES        = 20     # Coincide con training
MAX_NEW_TOKENS      = 8192   # Suficiente para esquema comprimido
THRESHOLD_REVISION  = 0.75   # confidence_real bajo este valor → REQUIERE_REVISION
OUTPUT_DIR          = "/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_8b_v2"
MODEL_PATH          = "/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_8b_v2/final"
DATASET_PATH        = "/home/nvidia-ott/lsdc/cnee-native/dataset/dataset_FINAL_rev_100casos_v6.json"
BASE                = "/home/nvidia-ott/lsdc/cnee-native"

# ══════════════════════════════════════════
# 1. Cargar modelo
# ══════════════════════════════════════════
print("Cargando modelo...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_PATH,
    load_in_4bit=False,
    use_gradient_checkpointing=False,
)
MODEL_MAX_SEQ_LENGTH = getattr(model, "max_seq_length", 32768)
try:
    tokenizer.model_max_length = MODEL_MAX_SEQ_LENGTH
except Exception:
    setattr(tokenizer, "model_max_length", MODEL_MAX_SEQ_LENGTH)

print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")
print(f"MAX_IMAGENES: {MAX_IMAGENES} | MAX_NEW_TOKENS: {MAX_NEW_TOKENS} | "
      f"THRESHOLD_REVISION: {THRESHOLD_REVISION}")

# ══════════════════════════════════════════
# 2. Cargar dataset y crear split
# ══════════════════════════════════════════
print("\nCargando dataset...")
with open(DATASET_PATH) as f:
    data = json.load(f)

casos = data["casos"]

random.seed(42)
grupos = defaultdict(list)
for i, caso in enumerate(casos):
    grupos[caso["metadata"]["label"]].append(i)

val_indices = []
for label, indices in grupos.items():
    n_val = max(1, int(len(indices) * 0.1))
    val_indices.extend(random.sample(indices, n_val))

print(f"Casos en validacion: {len(val_indices)}")

# ══════════════════════════════════════════
# 3. Funciones auxiliares
# ══════════════════════════════════════════
print("\n=== Evaluacion del modelo en set de validacion ===")
FastVisionModel.for_inference(model)

eval_casos = [casos[i] for i in val_indices]


def extraer_decision(texto):
    """Extrae la decision APROBADO/RECHAZADO del JSON generado."""
    match = re.search(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"', texto)
    if match:
        return match.group(1)
    if "APROBADO" in texto.upper():
        return "APROBADO"
    if "RECHAZADO" in texto.upper():
        return "RECHAZADO"
    return "DESCONOCIDO"


def calcular_decision_por_reglas(prediccion_texto):
    """
    Aplica las reglas del prompt CNEE para determinar decision y confidence reales.

    REGLAS:
      - causa_fuerza_mayor.cumple = false      → RECHAZADO (conf 0.90)
      - causa_FM=true Y cumplidos = 7          → APROBADO (conf 0.95)
      - causa_FM=true Y cumplidos = 6          → APROBADO (conf 0.82)
      - causa_FM=true Y cumplidos = 5          → APROBADO (conf 0.65, zona gris)
      - causa_FM=true Y cumplidos < 5          → RECHAZADO (conf 0.55, dudoso)
      - JSON invalido o sin estructura         → INDETERMINADO

    ESCALACION:
      - confidence_real < THRESHOLD_REVISION   → REQUIERE_REVISION_MANUAL
    """
    try:
        parsed = json.loads(prediccion_texto)
    except (json.JSONDecodeError, TypeError):
        return {
            'decision_regla':           'INDETERMINADO',
            'confidence_real':          0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple':          None,
            'decision_final':           'REQUIERE_REVISION_MANUAL',
            'motivo':                   'json_invalido_o_truncado',
        }

    criterios = parsed.get('criterios', {})
    if not criterios:
        return {
            'decision_regla':           'INDETERMINADO',
            'confidence_real':          0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple':          None,
            'decision_final':           'REQUIERE_REVISION_MANUAL',
            'motivo':                   'sin_criterios',
        }

    # Re-contar (no confiar en el campo del modelo)
    criterios_cumplidos_real = sum(
        1 for c in criterios.values() if c.get('cumple') is True
    )
    causa_fm = criterios.get('causa_fuerza_mayor', {})
    causa_fm_cumple = causa_fm.get('cumple')

    # ── Aplicar reglas ──
    if causa_fm_cumple is False:
        decision_regla = 'RECHAZADO'
        confidence_real = 0.90
        motivo = 'causa_FM_no_cumple'

    elif causa_fm_cumple is True and criterios_cumplidos_real >= 5:
        decision_regla = 'APROBADO'
        if criterios_cumplidos_real == 7:
            confidence_real = 0.95
            motivo = 'aprobado_7de7'
        elif criterios_cumplidos_real == 6:
            confidence_real = 0.82
            motivo = 'aprobado_6de7'
        else:  # 5
            confidence_real = 0.65
            motivo = 'aprobado_5de7_zona_gris'

    elif causa_fm_cumple is True and criterios_cumplidos_real < 5:
        decision_regla = 'RECHAZADO'
        confidence_real = 0.55
        motivo = 'rechazado_causa_ok_pero_docs_insuficientes'

    else:
        decision_regla = 'INDETERMINADO'
        confidence_real = 0.0
        motivo = 'causa_FM_no_determinable'

    # ── Escalacion ──
    if decision_regla == 'INDETERMINADO':
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final = motivo
    elif confidence_real < THRESHOLD_REVISION:
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final = f'{motivo}_confidence_bajo'
    else:
        decision_final = decision_regla
        motivo_final = motivo

    return {
        'decision_regla':           decision_regla,
        'confidence_real':          round(confidence_real, 2),
        'criterios_cumplidos_real': criterios_cumplidos_real,
        'causa_fm_cumple':          causa_fm_cumple,
        'decision_final':           decision_final,
        'motivo':                   motivo_final,
    }


def inferir_caso(caso):
    """Ejecuta inferencia. Retorna: (texto_generado, n_tokens, truncado)"""
    imagenes = caso["images"][:MAX_IMAGENES]
    rutas    = [f"{BASE}/{img}" for img in imagenes]

    texto_user = ""
    for item in caso["messages"][0]["content"]:
        if item.get("type") == "text" and item.get("text"):
            texto_user = item["text"]

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": ruta} for ruta in rutas] +
                   [{"type": "text",  "text": texto_user}]
    }]

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

    n_input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    n_generated = output_ids.shape[1] - n_input_tokens
    last_token = output_ids[0, -1].item()
    truncado = (last_token != tokenizer.eos_token_id and n_generated >= MAX_NEW_TOKENS)

    generated = tokenizer.decode(
        output_ids[0][n_input_tokens:],
        skip_special_tokens=True
    )
    return generated, n_generated, truncado


# ══════════════════════════════════════════
# 4. Loop de evaluacion
# ══════════════════════════════════════════
resultados = []
tiempos_casos = []
total_inicio = time.time()

for i, caso in enumerate(eval_casos):
    caso_inicio = time.time()
    print(f"\n  Caso {i+1}/{len(eval_casos)}: {caso['id']}...")

    ground_truth = ""
    for item in caso["messages"][1]["content"]:
        if item.get("type") == "text" and item.get("text"):
            ground_truth = item["text"]

    label_real = caso["metadata"]["label"]
    prediccion, n_tokens, truncado = inferir_caso(caso)
    tiempo_caso = time.time() - caso_inicio

    # Decision del modelo (lectura directa)
    label_pred_modelo = extraer_decision(prediccion)
    vqa_ok_modelo = (label_pred_modelo == label_real)
    exact = (ground_truth.strip() == prediccion.strip())

    # Decision por reglas + escalacion
    analisis = calcular_decision_por_reglas(prediccion)
    decision_final = analisis['decision_final']

    # Clasificar resultado final
    if decision_final == 'REQUIERE_REVISION_MANUAL':
        estado = 'ESC'
        es_correcto_auto = None  # No aplica, se escalo
    elif decision_final == label_real:
        estado = 'OK'
        es_correcto_auto = True
    else:
        estado = 'FAIL'
        es_correcto_auto = False

    tiempos_casos.append(tiempo_caso)

    print(f"    Real: {label_real:<10} | Modelo: {label_pred_modelo:<11} | "
          f"Final: {decision_final:<25} | C: {analisis['criterios_cumplidos_real']}/7 | "
          f"Conf: {analisis['confidence_real']:.2f} | {estado} | "
          f"Tokens: {n_tokens:>4} | Tiempo: {tiempo_caso:.1f}s")

    resultados.append({
        "caso":                     caso["id"],
        "label_real":               label_real,
        "label_pred_modelo":        label_pred_modelo,
        "decision_regla":           analisis['decision_regla'],
        "decision_final":           decision_final,
        "confidence_real":          analisis['confidence_real'],
        "criterios_cumplidos_real": analisis['criterios_cumplidos_real'],
        "causa_fm_cumple":          analisis['causa_fm_cumple'],
        "motivo":                   analisis['motivo'],
        "vqa_ok_modelo":            vqa_ok_modelo,
        "es_correcto_auto":         es_correcto_auto,
        "exact_match":              exact,
        "prediccion":               prediccion,
        "tokens_generados":         n_tokens,
        "truncado":                 truncado,
        "tiempo_s":                 round(tiempo_caso, 2),
    })

tiempo_total = time.time() - total_inicio

# ══════════════════════════════════════════
# 5. Calculo de metricas
# ══════════════════════════════════════════
n_val = len(eval_casos)

# ── Metricas del MODELO ORIGINAL (sin reglas) ──
vqa_correct_modelo = sum(1 for r in resultados if r["vqa_ok_modelo"])
exact_matches = sum(1 for r in resultados if r["exact_match"])

tp_m = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["label_pred_modelo"]=="APROBADO")
tn_m = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["label_pred_modelo"]=="RECHAZADO")
fp_m = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["label_pred_modelo"]=="APROBADO")
fn_m = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["label_pred_modelo"]=="RECHAZADO")

prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
rec_m  = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
f1_m   = 2 * prec_m * rec_m / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0

# ── Metricas del SISTEMA COMPLETO (con reglas + escalacion) ──
n_aprobado_auto  = sum(1 for r in resultados if r["decision_final"] == "APROBADO")
n_rechazado_auto = sum(1 for r in resultados if r["decision_final"] == "RECHAZADO")
n_escalados     = sum(1 for r in resultados if r["decision_final"] == "REQUIERE_REVISION_MANUAL")

correctos_auto   = sum(1 for r in resultados if r["es_correcto_auto"] is True)
incorrectos_auto = sum(1 for r in resultados if r["es_correcto_auto"] is False)

n_automaticos = n_val - n_escalados

# Metricas confusion matrix sobre solo los automaticos
tp_s = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["decision_final"]=="APROBADO")
tn_s = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["decision_final"]=="RECHAZADO")
fp_s = sum(1 for r in resultados if r["label_real"]=="RECHAZADO" and r["decision_final"]=="APROBADO")
fn_s = sum(1 for r in resultados if r["label_real"]=="APROBADO"  and r["decision_final"]=="RECHAZADO")

prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
rec_s  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0

# Stats auxiliares
tokens_lista = [r["tokens_generados"] for r in resultados]
n_truncados  = sum(1 for r in resultados if r["truncado"])

# ══════════════════════════════════════════
# 6. Reporte en consola
# ══════════════════════════════════════════
print(f"\n{'='*65}")
print(f"METRICAS DEL MODELO ORIGINAL (sin reglas)")
print(f"{'='*65}")
print(f"  VQA Accuracy:   {vqa_correct_modelo}/{n_val}  ({vqa_correct_modelo/n_val*100:.1f}%)")
print(f"  Precision:      {prec_m:.3f}")
print(f"  Recall:         {rec_m:.3f}")
print(f"  F1-Score:       {f1_m:.3f}")
print(f"  TP={tp_m}  TN={tn_m}  FP={fp_m}  FN={fn_m}")

print(f"\n{'='*65}")
print(f"METRICAS DEL SISTEMA HIBRIDO (modelo + reglas + escalacion)")
print(f"{'='*65}")
print(f"  Distribucion de decisiones finales:")
print(f"    APROBADO automatico:     {n_aprobado_auto}/{n_val}  "
      f"({n_aprobado_auto/n_val*100:.1f}%)")
print(f"    RECHAZADO automatico:    {n_rechazado_auto}/{n_val}  "
      f"({n_rechazado_auto/n_val*100:.1f}%)")
print(f"    REQUIERE_REVISION:       {n_escalados}/{n_val}  "
      f"({n_escalados/n_val*100:.1f}%)")
print()
print(f"  De los {n_automaticos} casos automaticos:")
if n_automaticos > 0:
    print(f"    Correctos:    {correctos_auto}/{n_automaticos}  "
          f"({correctos_auto/n_automaticos*100:.1f}%)")
    print(f"    Incorrectos:  {incorrectos_auto}/{n_automaticos}  "
          f"({incorrectos_auto/n_automaticos*100:.1f}%)")
print()
print(f"  Indicadores de produccion:")
print(f"    Tasa automatizacion segura:  {correctos_auto/n_val*100:.1f}%")
print(f"    Tasa falsos positivos auto:  {incorrectos_auto/n_val*100:.1f}%")
print(f"    Tasa escalacion humana:      {n_escalados/n_val*100:.1f}%")
print()
print(f"  Confusion matrix (solo automaticos):")
print(f"    TP={tp_s}  TN={tn_s}  FP={fp_s}  FN={fn_s}")
print(f"    Precision: {prec_s:.3f}  Recall: {rec_s:.3f}  F1: {f1_s:.3f}")

print(f"\n{'─'*65}")
print(f"TIEMPOS DE INFERENCIA")
print(f"{'─'*65}")
print(f"  Tiempo total:     {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
print(f"  Tiempo promedio:  {tiempo_total/n_val:.1f}s/caso")
print(f"  Caso mas rapido:  {min(tiempos_casos):.1f}s")
print(f"  Caso mas lento:   {max(tiempos_casos):.1f}s")

print(f"\n{'─'*65}")
print(f"TOKENS GENERADOS")
print(f"{'─'*65}")
print(f"  Min:     {min(tokens_lista)}")
print(f"  Media:   {int(np.mean(tokens_lista))}")
print(f"  P50:     {int(np.median(tokens_lista))}")
print(f"  P95:     {int(np.percentile(tokens_lista, 95))}")
print(f"  Max:     {max(tokens_lista)}")
print(f"  Truncados: {n_truncados}/{n_val}")
print(f"{'='*65}")

# ══════════════════════════════════════════
# 7. Guardar JSON
# ══════════════════════════════════════════
metricas_output = {
    "config": {
        "max_imagenes":       MAX_IMAGENES,
        "max_new_tokens":     MAX_NEW_TOKENS,
        "threshold_revision": THRESHOLD_REVISION,
        "model_path":         MODEL_PATH,
        "dataset_path":       DATASET_PATH,
    },
    "n_val": n_val,
    "metricas_modelo_original": {
        "vqa_accuracy":     vqa_correct_modelo / n_val * 100,
        "exact_match":      exact_matches / n_val * 100,
        "precision":        prec_m,
        "recall":           rec_m,
        "f1":               f1_m,
        "confusion_matrix": {"tp": tp_m, "tn": tn_m, "fp": fp_m, "fn": fn_m},
    },
    "metricas_sistema_hibrido": {
        "n_aprobado_auto":          n_aprobado_auto,
        "n_rechazado_auto":         n_rechazado_auto,
        "n_escalados":              n_escalados,
        "correctos_auto":           correctos_auto,
        "incorrectos_auto":         incorrectos_auto,
        "tasa_automatizacion_segura": round(correctos_auto / n_val * 100, 2),
        "tasa_falsos_positivos_auto": round(incorrectos_auto / n_val * 100, 2),
        "tasa_escalacion":          round(n_escalados / n_val * 100, 2),
        "precision_auto":           prec_s,
        "recall_auto":              rec_s,
        "f1_auto":                  f1_s,
        "confusion_matrix_auto":    {"tp": tp_s, "tn": tn_s, "fp": fp_s, "fn": fn_s},
    },
    "tiempos": {
        "tiempo_total_s":    round(tiempo_total, 2),
        "tiempo_promedio_s": round(tiempo_total / n_val, 2),
        "tiempo_minimo_s":   round(min(tiempos_casos), 2),
        "tiempo_maximo_s":   round(max(tiempos_casos), 2),
    },
    "tokens": {
        "min":           int(min(tokens_lista)),
        "media":         int(np.mean(tokens_lista)),
        "p50":           int(np.median(tokens_lista)),
        "p95":           int(np.percentile(tokens_lista, 95)),
        "max":           int(max(tokens_lista)),
        "n_truncados":   int(n_truncados),
        "pct_truncados": round(n_truncados / n_val * 100, 2),
    },
    "resultados": resultados,
}
with open(f"{OUTPUT_DIR}/metricas_validacion.json", "w", encoding="utf-8") as f:
    json.dump(metricas_output, f, ensure_ascii=False, indent=2)
print(f"\nMetricas guardadas en: {OUTPUT_DIR}/metricas_validacion.json")

# ══════════════════════════════════════════
# 8. Grafica de evaluacion (2x2 = 4 paneles)
# ══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Evaluacion del Modelo Fine-tuned + Sistema Hibrido — Qwen3-VL 8B (CNEE)",
             fontsize=13, fontweight="bold")

# Plot 1: Comparacion modelo vs sistema hibrido
ax = axes[0, 0]
labels = ['VQA\nAccuracy', 'Precision', 'Recall', 'F1-Score']
vals_modelo = [vqa_correct_modelo/n_val*100, prec_m*100, rec_m*100, f1_m*100]
vals_sistema = [correctos_auto/n_val*100, prec_s*100, rec_s*100, f1_s*100]

x = np.arange(len(labels))
width = 0.35
b1 = ax.bar(x - width/2, vals_modelo, width, label='Modelo solo', color='#0D1B3E')
b2 = ax.bar(x + width/2, vals_sistema, width, label='Sistema hibrido (auto)', color='#16A34A')
ax.set_title("Metricas: Modelo vs Sistema con Escalacion")
ax.set_ylabel("Porcentaje (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 120)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2, f"{h:.1f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: Distribucion de decisiones finales
ax = axes[0, 1]
cats = ['APROBADO\nautomatico', 'RECHAZADO\nautomatico', 'REQUIERE\nREVISION']
vals = [n_aprobado_auto, n_rechazado_auto, n_escalados]
cols = ['#16A34A', '#0D1B3E', '#F59E0B']
bars = ax.bar(cats, vals, color=cols, width=0.6)
ax.set_title("Distribucion de Decisiones Finales del Sistema")
ax.set_ylabel("Cantidad de casos")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, vals):
    pct = v / n_val * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{v}\n({pct:.0f}%)", ha='center', va='bottom',
            fontweight='bold', fontsize=11)

# Plot 3: Confusion matrix del sistema hibrido (solo automaticos)
ax = axes[1, 0]
confusion = [[tp_s, fn_s], [fp_s, tn_s]]
im = ax.imshow(confusion, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred APROBADO", "Pred RECHAZADO"])
ax.set_yticklabels(["Real APROBADO", "Real RECHAZADO"])
ax.set_title(f"Matriz de Confusion (solo {n_automaticos} automaticos)")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(confusion[i][j]),
                ha="center", va="center",
                fontsize=20, fontweight="bold",
                color="white" if confusion[i][j] > n_automaticos/4 else "black")

# Plot 4: Indicadores de produccion
ax = axes[1, 1]
indicadores = ['Auto\nSegura', 'Falsos+\nAuto', 'Escalacion\nHumana']
vals_ind = [
    correctos_auto / n_val * 100,
    incorrectos_auto / n_val * 100,
    n_escalados / n_val * 100,
]
cols_ind = ['#16A34A', '#DC2626', '#F59E0B']
bars = ax.bar(indicadores, vals_ind, color=cols_ind, width=0.6)
ax.set_title("Indicadores de Produccion")
ax.set_ylabel("Porcentaje (%)")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, vals_ind):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{v:.1f}%", ha='center', va='bottom',
            fontweight='bold', fontsize=11)

plt.tight_layout()
eval_plot_path = f"{OUTPUT_DIR}/evaluacion_modelo_rules.png"
plt.savefig(eval_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafica guardada en: {eval_plot_path}")

print("\n=== Evaluacion completada ===")