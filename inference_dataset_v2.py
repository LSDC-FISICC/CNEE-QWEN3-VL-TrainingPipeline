"""
Script de inferencia del modelo Qwen3-VL fine-tuned sobre casos NO vistos en training.
Usa dataset_inferencia_test_v2.json (sin respuesta del asistente).

Reporta metricas globales y por clase (APROBADO / RECHAZADO) usando label_real como
ground truth para medir generalizacion sobre datos completamente nuevos.

v2 — Mejoras:
  - repetition_penalty para evitar loops SCADA
  - Checkpoint / resume: guarda cada 5 casos, reanuda si el script se interrumpe
  - Liberacion explicita de tensores y cierre de imagenes PIL entre casos
  - ETA dinamico calculado con tiempos reales del run
"""

import json
import re
import os
import time
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from unsloth import FastVisionModel
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Inferencia Qwen3-VL v2 — Casos nuevos (no vistos en training) ===\n")

# ══════════════════════════════════════════
# Configuracion
# ══════════════════════════════════════════
MAX_IMAGENES        = 20
MAX_NEW_TOKENS      = 8192
THRESHOLD_REVISION  = 0.75
OUTPUT_DIR          = "/home/nvidia-ott/lsdc/cnee-native/output/inferencia_v4"
MODEL_PATH          = "/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_8b_v1/final"
DATASET_PATH        = "/home/nvidia-ott/lsdc/cnee-native/dataset/dataset_inferencia_train_v2.json"
BASE                = "/home/nvidia-ott/lsdc/cnee-native"
CHECKPOINT_INTERVAL = 5   # Guarda checkpoint cada N casos

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
print(f"MAX_IMAGENES={MAX_IMAGENES} | MAX_NEW_TOKENS={MAX_NEW_TOKENS} | "
      f"THRESHOLD_REVISION={THRESHOLD_REVISION}")

# ══════════════════════════════════════════
# 2. Cargar dataset de inferencia
# ══════════════════════════════════════════
print("\nCargando dataset de inferencia...")
with open(DATASET_PATH) as f:
    data = json.load(f)

casos   = data["casos"]
n_total = len(casos)
print(f"Total casos: {n_total}")
print(f"  Aprobados (ground truth): {data['dataset_info']['aprobados']}")
print(f"  Rechazados (ground truth): {data['dataset_info']['rechazados']}")

# ══════════════════════════════════════════
# 3. Checkpoint — carga resultados previos
# ══════════════════════════════════════════
CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoint_inferencia.json"
resultados      = []
casos_procesados = set()

if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH) as f:
        resultados = json.load(f)
    casos_procesados = {r["caso"] for r in resultados}
    print(f"\nCheckpoint encontrado: {len(resultados)}/{n_total} casos ya procesados.")
    print(f"Reanudando desde el caso {len(resultados) + 1}...")
else:
    print("\nNo hay checkpoint previo. Iniciando desde cero.")

# ══════════════════════════════════════════
# 4. Funciones auxiliares
# ══════════════════════════════════════════
FastVisionModel.for_inference(model)


def extraer_decision(texto):
    match = re.search(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"', texto)
    if match:
        return match.group(1)
    if "APROBADO" in texto.upper():
        return "APROBADO"
    if "RECHAZADO" in texto.upper():
        return "RECHAZADO"
    return "DESCONOCIDO"


def calcular_decision_por_reglas(prediccion_texto):
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

    criterios_cumplidos_real = sum(
        1 for c in criterios.values() if c.get('cumple') is True
    )
    causa_fm        = criterios.get('causa_fuerza_mayor', {})
    causa_fm_cumple = causa_fm.get('cumple')

    if causa_fm_cumple is False:
        decision_regla  = 'RECHAZADO'
        confidence_real = 0.90
        motivo          = 'causa_FM_no_cumple'
    elif causa_fm_cumple is True and criterios_cumplidos_real >= 5:
        decision_regla = 'APROBADO'
        if criterios_cumplidos_real == 7:
            confidence_real = 0.95
            motivo          = 'aprobado_7de7'
        elif criterios_cumplidos_real == 6:
            confidence_real = 0.82
            motivo          = 'aprobado_6de7'
        else:
            confidence_real = 0.65
            motivo          = 'aprobado_5de7_zona_gris'
    elif causa_fm_cumple is True and criterios_cumplidos_real < 5:
        decision_regla  = 'RECHAZADO'
        confidence_real = 0.55
        motivo          = 'rechazado_causa_ok_pero_docs_insuficientes'
    else:
        decision_regla  = 'INDETERMINADO'
        confidence_real = 0.0
        motivo          = 'causa_FM_no_determinable'

    if decision_regla == 'INDETERMINADO':
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final   = motivo
    elif confidence_real < THRESHOLD_REVISION:
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final   = f'{motivo}_confidence_bajo'
    else:
        decision_final = decision_regla
        motivo_final   = motivo

    return {
        'decision_regla':           decision_regla,
        'confidence_real':          round(confidence_real, 2),
        'criterios_cumplidos_real': criterios_cumplidos_real,
        'causa_fm_cumple':          causa_fm_cumple,
        'decision_final':           decision_final,
        'motivo':                   motivo_final,
    }


def inferir_caso(caso):
    """Ejecuta inferencia. Retorna (texto, n_tokens, truncado)."""
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
            repetition_penalty=1.05,  # Penaliza tokens repetidos
            pad_token_id=tokenizer.eos_token_id,
        )

    n_generated = output_ids.shape[1] - n_input_tokens
    last_token  = output_ids[0, -1].item()
    truncado    = (last_token != tokenizer.eos_token_id and n_generated >= MAX_NEW_TOKENS)

    generated = tokenizer.decode(
        output_ids[0][n_input_tokens:],
        skip_special_tokens=True
    )

    # Liberar tensores y cerrar imagenes para evitar acumulacion en memoria
    del inputs, output_ids
    torch.cuda.empty_cache()
    for img in images:
        img.close()

    return generated, n_generated, truncado


# ══════════════════════════════════════════
# 5. Loop de inferencia
# ══════════════════════════════════════════
tiempos_casos = [r["tiempo_s"] for r in resultados]  # Incluye tiempos previos si hay checkpoint
total_inicio  = time.time()
casos_nuevos  = 0

print(f"\nInferencia sobre {n_total - len(casos_procesados)} casos pendientes.\n")

for i, caso in enumerate(casos):

    # Skip si ya fue procesado en un run anterior
    if caso["id"] in casos_procesados:
        print(f"  [{i+1:3d}/{n_total}] {caso['id']} — skipped (checkpoint)")
        continue

    caso_inicio = time.time()
    label_real  = caso["label_real"]

    print(f"  [{i+1:3d}/{n_total}] {caso['id']}...", end='', flush=True)

    prediccion, n_tokens, truncado = inferir_caso(caso)
    tiempo_caso = time.time() - caso_inicio

    label_pred_modelo = extraer_decision(prediccion)
    vqa_ok_modelo     = (label_pred_modelo == label_real)

    analisis       = calcular_decision_por_reglas(prediccion)
    decision_final = analisis['decision_final']

    if decision_final == 'REQUIERE_REVISION_MANUAL':
        estado          = 'ESC'
        es_correcto_auto = None
    elif decision_final == label_real:
        estado          = 'OK'
        es_correcto_auto = True
    else:
        estado          = 'FAIL'
        es_correcto_auto = False

    tiempos_casos.append(tiempo_caso)
    casos_nuevos += 1

    # ETA dinamico basado en tiempos reales
    casos_restantes = n_total - (i + 1)
    avg_tiempo      = sum(tiempos_casos) / len(tiempos_casos)
    eta_min         = casos_restantes * avg_tiempo / 60

    print(f" Real:{label_real[:4]:<4} Modelo:{label_pred_modelo[:4]:<4} "
          f"Final:{decision_final[:4]:<4} C:{analisis['criterios_cumplidos_real']}/7 "
          f"Conf:{analisis['confidence_real']:.2f} {estado} "
          f"Tok:{n_tokens:>4} {tiempo_caso:.0f}s | ETA:{eta_min:.0f}min")

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
        "prediccion":               prediccion,
        "tokens_generados":         n_tokens,
        "truncado":                 truncado,
        "tiempo_s":                 round(tiempo_caso, 2),
    })

    # Checkpoint cada N casos
    if casos_nuevos % CHECKPOINT_INTERVAL == 0:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False)
        print(f"  >>> Checkpoint guardado ({len(resultados)}/{n_total})")

tiempo_total = time.time() - total_inicio

# ══════════════════════════════════════════
# 6. Metricas
# ══════════════════════════════════════════
def calcular_metricas(subset):
    n = len(subset)
    if n == 0:
        return None

    vqa_modelo = sum(1 for r in subset if r["vqa_ok_modelo"])

    tp_m = sum(1 for r in subset if r["label_real"]=="APROBADO"  and r["label_pred_modelo"]=="APROBADO")
    tn_m = sum(1 for r in subset if r["label_real"]=="RECHAZADO" and r["label_pred_modelo"]=="RECHAZADO")
    fp_m = sum(1 for r in subset if r["label_real"]=="RECHAZADO" and r["label_pred_modelo"]=="APROBADO")
    fn_m = sum(1 for r in subset if r["label_real"]=="APROBADO"  and r["label_pred_modelo"]=="RECHAZADO")

    prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0
    rec_m  = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0
    f1_m   = 2 * prec_m * rec_m / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0

    n_apr  = sum(1 for r in subset if r["decision_final"] == "APROBADO")
    n_rch  = sum(1 for r in subset if r["decision_final"] == "RECHAZADO")
    n_esc  = sum(1 for r in subset if r["decision_final"] == "REQUIERE_REVISION_MANUAL")
    corr   = sum(1 for r in subset if r["es_correcto_auto"] is True)
    incorr = sum(1 for r in subset if r["es_correcto_auto"] is False)

    tp_s = sum(1 for r in subset if r["label_real"]=="APROBADO"  and r["decision_final"]=="APROBADO")
    tn_s = sum(1 for r in subset if r["label_real"]=="RECHAZADO" and r["decision_final"]=="RECHAZADO")
    fp_s = sum(1 for r in subset if r["label_real"]=="RECHAZADO" and r["decision_final"]=="APROBADO")
    fn_s = sum(1 for r in subset if r["label_real"]=="APROBADO"  and r["decision_final"]=="RECHAZADO")

    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
    rec_s  = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    f1_s   = 2 * prec_s * rec_s / (prec_s + rec_s) if (prec_s + rec_s) > 0 else 0

    return {
        'n':                  n,
        'vqa_modelo':         vqa_modelo,
        'vqa_pct':            vqa_modelo / n * 100,
        'precision_m':        prec_m,
        'recall_m':           rec_m,
        'f1_m':               f1_m,
        'confusion_m':        {"tp": tp_m, "tn": tn_m, "fp": fp_m, "fn": fn_m},
        'n_aprobado':         n_apr,
        'n_rechazado':        n_rch,
        'n_escalado':         n_esc,
        'correctos_auto':     corr,
        'incorrectos_auto':   incorr,
        'tasa_auto_segura':   corr / n * 100,
        'tasa_falsos_auto':   incorr / n * 100,
        'tasa_escalacion':    n_esc / n * 100,
        'precision_s':        prec_s,
        'recall_s':           rec_s,
        'f1_s':               f1_s,
        'confusion_s':        {"tp": tp_s, "tn": tn_s, "fp": fp_s, "fn": fn_s},
    }


m     = calcular_metricas(resultados)
m_apr = calcular_metricas([r for r in resultados if r["label_real"] == "APROBADO"])
m_rch = calcular_metricas([r for r in resultados if r["label_real"] == "RECHAZADO"])

# ══════════════════════════════════════════
# 7. Imprimir resultados
# ══════════════════════════════════════════
def imprimir_metricas(m, titulo):
    print(f"\n{'='*70}")
    print(f"  {titulo}  (n={m['n']})")
    print(f"{'='*70}")
    print(f"  MODELO ORIGINAL:")
    print(f"    VQA Accuracy: {m['vqa_modelo']}/{m['n']} ({m['vqa_pct']:.1f}%)  "
          f"| Precision: {m['precision_m']:.3f}  Recall: {m['recall_m']:.3f}  F1: {m['f1_m']:.3f}")
    print(f"    Confusion: TP={m['confusion_m']['tp']} TN={m['confusion_m']['tn']} "
          f"FP={m['confusion_m']['fp']} FN={m['confusion_m']['fn']}")
    print(f"  SISTEMA HIBRIDO (modelo + reglas + escalacion):")
    print(f"    APROBADO auto:   {m['n_aprobado']:>3} ({m['n_aprobado']/m['n']*100:.1f}%)")
    print(f"    RECHAZADO auto:  {m['n_rechazado']:>3} ({m['n_rechazado']/m['n']*100:.1f}%)")
    print(f"    ESCALADO:        {m['n_escalado']:>3} ({m['n_escalado']/m['n']*100:.1f}%)")
    print(f"    Tasa automatizacion segura: {m['tasa_auto_segura']:.1f}%")
    print(f"    Tasa falsos positivos auto: {m['tasa_falsos_auto']:.1f}%")
    print(f"    Confusion (auto): TP={m['confusion_s']['tp']} TN={m['confusion_s']['tn']} "
          f"FP={m['confusion_s']['fp']} FN={m['confusion_s']['fn']}  "
          f"P={m['precision_s']:.3f} R={m['recall_s']:.3f} F1={m['f1_s']:.3f}")


imprimir_metricas(m,     "TOTAL (casos nuevos, no vistos en training)")
imprimir_metricas(m_apr, "APROBADOS (ground truth)")
imprimir_metricas(m_rch, "RECHAZADOS (ground truth)")

# ══════════════════════════════════════════
# 8. Fallos
# ══════════════════════════════════════════
print(f"\n{'='*70}")
print(f"  CASOS DONDE EL MODELO FALLA (sin reglas)")
print(f"{'='*70}")
fallos_modelo = [r for r in resultados if not r["vqa_ok_modelo"]]
print(f"  Total fallos modelo: {len(fallos_modelo)}/{n_total}")
for r in fallos_modelo:
    print(f"    {r['caso']}: real={r['label_real']:<10} "
          f"pred={r['label_pred_modelo']:<12} criterios={r['criterios_cumplidos_real']}/7  "
          f"motivo={r['motivo']}")

print(f"\n{'='*70}")
print(f"  CASOS DONDE EL SISTEMA HIBRIDO FALLA AUTOMATICAMENTE")
print(f"{'='*70}")
fallos_sistema = [r for r in resultados if r["es_correcto_auto"] is False]
print(f"  Total fallos automaticos: {len(fallos_sistema)}/{n_total}")
for r in fallos_sistema:
    print(f"    {r['caso']}: real={r['label_real']:<10} "
          f"final={r['decision_final']:<12} conf={r['confidence_real']:.2f}  "
          f"motivo={r['motivo']}")

# ══════════════════════════════════════════
# 9. Tiempos y tokens
# ══════════════════════════════════════════
tokens_lista = [r["tokens_generados"] for r in resultados]
n_truncados  = sum(1 for r in resultados if r["truncado"])

print(f"\n{'='*70}")
print(f"  TIEMPOS Y TOKENS")
print(f"{'='*70}")
print(f"  Tiempo total sesion: {tiempo_total:.0f}s ({tiempo_total/60:.1f} min)")
print(f"  Promedio/caso:       {sum(tiempos_casos)/len(tiempos_casos):.1f}s")
print(f"  Tokens: min={min(tokens_lista)} media={int(np.mean(tokens_lista))} "
      f"P50={int(np.median(tokens_lista))} P95={int(np.percentile(tokens_lista,95))} "
      f"max={max(tokens_lista)}")
print(f"  Truncados: {n_truncados}/{n_total} ({n_truncados/n_total*100:.1f}%)")

# ══════════════════════════════════════════
# 10. Guardar JSON final y limpiar checkpoint
# ══════════════════════════════════════════
output = {
    "config": {
        "max_imagenes":           MAX_IMAGENES,
        "max_new_tokens":         MAX_NEW_TOKENS,
        "threshold_revision":     THRESHOLD_REVISION,
        "no_repeat_ngram_size":   4,
        "repetition_penalty":     1.15,
        "model_path":             MODEL_PATH,
        "dataset_path":           DATASET_PATH,
    },
    "n_total":            n_total,
    "metricas_total":     m,
    "metricas_aprobados": m_apr,
    "metricas_rechazados":m_rch,
    "tiempos": {
        "tiempo_total_s":    round(tiempo_total, 2),
        "tiempo_promedio_s": round(sum(tiempos_casos) / len(tiempos_casos), 2),
        "tiempo_minimo_s":   round(min(tiempos_casos), 2),
        "tiempo_maximo_s":   round(max(tiempos_casos), 2),
    },
    "tokens": {
        "min":         int(min(tokens_lista)),
        "media":       int(np.mean(tokens_lista)),
        "p50":         int(np.median(tokens_lista)),
        "p95":         int(np.percentile(tokens_lista, 95)),
        "max":         int(max(tokens_lista)),
        "n_truncados": int(n_truncados),
    },
    "resultados": resultados,
}

output_path = f"{OUTPUT_DIR}/resultados_inferencia.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nResultados guardados en: {output_path}")

# Elimina checkpoint al completar exitosamente
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint eliminado (run completo).")

# ══════════════════════════════════════════
# 11. Grafica
# ══════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle(f"Inferencia — Casos nuevos (n={n_total}) — Qwen3-VL 8B (CNEE)",
             fontsize=13, fontweight="bold")

# Plot 1: Accuracy modelo vs sistema por clase
ax = axes[0, 0]
labels   = ['APROBADOS\n(modelo)', 'APROBADOS\n(sistema)', 'RECHAZADOS\n(modelo)', 'RECHAZADOS\n(sistema)']
vals_acc = [
    m_apr['vqa_pct'],
    m_apr['tasa_auto_segura'],
    m_rch['vqa_pct'],
    m_rch['tasa_auto_segura'],
]
vals_fp = [
    m_apr['confusion_m']['fp'] / m_apr['n'] * 100,
    m_apr['tasa_falsos_auto'],
    m_rch['confusion_m']['fp'] / m_rch['n'] * 100,
    m_rch['tasa_falsos_auto'],
]
x  = np.arange(len(labels))
w  = 0.35
b1 = ax.bar(x - w/2, vals_acc, w, label='Accuracy / Auto correcto', color='#16A34A')
b2 = ax.bar(x + w/2, vals_fp,  w, label='Falsos positivos',         color='#DC2626')
ax.set_title("Accuracy vs Falsos Positivos por Clase")
ax.set_ylabel("Porcentaje (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis="y")
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 2: Distribucion de decisiones finales
ax   = axes[0, 1]
cats = ['APROBADO\nauto', 'RECHAZADO\nauto', 'REQUIERE\nREVISION']
vals = [m['n_aprobado'], m['n_rechazado'], m['n_escalado']]
cols = ['#16A34A', '#0D1B3E', '#F59E0B']
bars = ax.bar(cats, vals, color=cols, width=0.6)
ax.set_title(f"Distribucion de Decisiones Finales (n={n_total})")
ax.set_ylabel("Cantidad de casos")
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, vals):
    pct = v / n_total * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v}\n({pct:.0f}%)", ha='center', va='bottom',
            fontweight='bold', fontsize=11)

# Plot 3: Confusion matrix sistema
ax     = axes[1, 0]
n_auto = m['n'] - m['n_escalado']
confusion = [[m['confusion_s']['tp'], m['confusion_s']['fn']],
             [m['confusion_s']['fp'], m['confusion_s']['tn']]]
ax.imshow(confusion, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred APROBADO", "Pred RECHAZADO"])
ax.set_yticklabels(["Real APROBADO", "Real RECHAZADO"])
ax.set_title(f"Matriz de Confusion — Sistema (auto, n={n_auto})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(confusion[i][j]),
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="white" if confusion[i][j] > n_auto / 4 else "black")

# Plot 4: Indicadores globales
ax       = axes[1, 1]
indicadores = ['VQA\nAccuracy', 'Auto\nSegura', 'Falsos+\nAuto', 'Escalacion']
vals_bar    = [m['vqa_pct'], m['tasa_auto_segura'], m['tasa_falsos_auto'], m['tasa_escalacion']]
cols_bar    = ['#16A34A', '#16A34A', '#DC2626', '#F59E0B']
bars        = ax.bar(indicadores, vals_bar, color=cols_bar, width=0.5)
ax.set_title("Indicadores Globales — Casos Nuevos")
ax.set_ylabel("Porcentaje (%)")
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis="y")
for bar, v in zip(bars, vals_bar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.1f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plot_path = f"{OUTPUT_DIR}/inferencia_resultados.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Grafica guardada en: {plot_path}")

print("\n=== Inferencia completada ===")