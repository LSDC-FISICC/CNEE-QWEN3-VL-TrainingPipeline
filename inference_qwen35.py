"""
Script de inferencia del modelo Qwen3.5 fine-tuned sobre casos NO vistos en training.
Adaptado desde la version Qwen3-VL.

Cambios clave respecto a la version Qwen3-VL:
  - FastVisionModel -> FastModel (API unificada de Unsloth)
  - apply_chat_template con enable_thinking=False (salida JSON directa)
  - limpiar_thinking() antes del parseo JSON — CRITICO para que las reglas funcionen
  - MODEL_PATH actualizado al checkpoint de Qwen3.5
  - max_seq_length explicito en from_pretrained

CAMBIOS v6 (correcciones post-analisis Dataset v8):
  A. AGREEMENT GATE: calcular_decision_por_reglas() recibe
     label_pred_modelo y nunca auto-decide en contra de la prediccion
     latente del modelo; el conflicto se escala como caso dificil
     (motivo 'conflicto_regla_vs_modelo').
  B. BUG FIX: no_repeat_ngram_size se declaraba en el config guardado
     pero NUNCA se pasaba a model.generate(). Ahora se pasa (valor 8).
  C. Parser robusto portado del script thinking: limpiar_thinking()
     maneja el tag </think> huerfano y extraer_json() recupera el JSON
     con regex cuando hay texto residual.
  D. Recuperacion de checkpoint: re-aplica parser + agreement gate a
     todos los casos y expulsa truncados/DESCONOCIDO para re-inferirlos.

CAMBIOS v7:
  - no_repeat_ngram_size REVERTIDO: rompe el boilerplate legitimo de los
    7 bloques de criterios y el parser deja de extraer 'criterios'.

CAMBIOS v9 (bug de la llave de cierre):
  I. BUG FIX CRITICO: el stopper disparaba con '"decision": "..."' sin exigir
     la llave de cierre. Como el chequeo corre cada STOP_CHECK_EVERY=8 tokens,
     en ~18% de los casos la generacion se cortaba UN TOKEN antes del '}' y el
     JSON quedaba inparseable -> 'json_invalido_o_truncado' -> escalacion
     forzada de casos perfectamente validos. Ahora DECISION_STOP_RE exige
     '"decision": "..." }'.
  J. Reparacion defensiva en extraer_json(): si el parseo falla, se balancean
     las llaves abiertas y se reintenta. Cubre cualquier corte residual.
  K. Contador STATS['json_reparados'] para observabilidad del fenomeno.

CAMBIOS v8 (loops ciclicos y escalacion directa):
  E. JsonCompleteOrLoopStop (StoppingCriteria): (a) corta la generacion
     cuando aparece el campo 'decision' (ultimo del schema desde v5,
     corte sin perdida); (b) aborta loops de conteo (timestamps
     incrementales), que repetition_penalty y no_repeat_ngram_size NO
     pueden frenar porque cada paso emite tokens nuevos.
  F. ESCALACION DIRECTA: salida truncada o loop abortado -> decision_final
     REQUIERE_REVISION_MANUAL de inmediato (motivo 'loop_ciclico_abortado'
     o 'salida_truncada'), sin pasar por reglas ni agreement gate.
  G. Extraccion ESTRICTA de la etiqueta en casos degenerados: solo el
     campo "decision"; el fallback por substring queda prohibido ahi
     porque inventa etiquetas desde las observaciones de un JSON a medias
     y contamina vqa_ok_modelo y el agreement gate.
  H. BUG FIX re-inferencia infinita: con decoding greedy un loop es
     determinista; re-inferir un caso en loop reproduce el mismo loop en
     cada reanudacion de checkpoint. Los casos 'loop_abortado' quedan
     como escalados FINALES y no se expulsan para re-inferencia. Los
     truncados legacy (checkpoints previos, sin el flag) se re-infieren
     UNA vez, ya protegidos por el loop guard.
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
from unsloth import FastModel                          # <-- antes FastVisionModel
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image

torch._dynamo.config.disable = True
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

print("=== Inferencia Qwen3.5 v8 — Casos nuevos (no vistos en training) ===\n")

# ══════════════════════════════════════════
# Configuracion
# ══════════════════════════════════════════
MAX_IMAGENES        = 24
MAX_NEW_TOKENS      = 8192
REPETITION_PENALTY  = 1.05
THRESHOLD_REVISION  = 0.75
CAP_ACREDITACION    = 0.60            # v10: cap when `acreditacion` is inconclusive
OUTPUT_DIR          = "/home/nvidia-ott/lsdc/cnee-native/output/inferencia_qwen35_9b_v11"
MODEL_PATH          = "/home/nvidia-ott/lsdc/cnee-native/output/qwen35_9b_v9/checkpoint-72"   # <-- checkpoint entrenado con dataset v9
DATASET_PATH        = "/home/nvidia-ott/lsdc/cnee-native/dataset/dataset_inferencia_full_v10.json"  # <-- prompt v10, paridad bit-a-bit con el training set
BASE                = "/home/nvidia-ott/lsdc/cnee-native"
CHECKPOINT_INTERVAL = 5
MAX_SEQ_LEN         = 8192                                                            # <-- fijo y explicito
ENABLE_THINKING     = False                                                          # <-- salida JSON directa

# ── v8: stopping-criteria config ─────────────────────────────────────
STOP_CHECK_EVERY  = 8      # run the decode check every N generated tokens
TS_WINDOW_TOKENS  = 200    # decode only this tail window for the loop check
TS_LOOP_THRESHOLD = 10     # >= this many HH:MM:SS in the window => counting loop
                           # (legitimate v9 output has at most ~5 in any window)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════
# 1. Cargar modelo
# ══════════════════════════════════════════
print("Cargando modelo...")
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    use_gradient_checkpointing=False,
)
try:
    tokenizer.model_max_length = MAX_SEQ_LEN
except Exception:
    setattr(tokenizer, "model_max_length", MAX_SEQ_LEN)

print(f"Modelo cargado. VRAM: {round(torch.cuda.memory_allocated()/1e9, 2)} GB")
print(f"MAX_IMAGENES={MAX_IMAGENES} | MAX_NEW_TOKENS={MAX_NEW_TOKENS} | "
      f"THRESHOLD_REVISION={THRESHOLD_REVISION} | ENABLE_THINKING={ENABLE_THINKING}")

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
FastModel.for_inference(model)

DECISION_RE  = re.compile(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"')
# v9: the STOP trigger additionally requires the closing brace, so generation
# never halts on a JSON object that has not been closed yet.
DECISION_STOP_RE = re.compile(r'"decision"\s*:\s*"(?:APROBADO|RECHAZADO)"\s*\}')
TIMESTAMP_RE = re.compile(r'\b\d{1,2}:\d{2}:\d{2}\b')

# v9: observability of the brace-repair path
STATS = {"json_reparados": 0}


class JsonCompleteOrLoopStop(StoppingCriteria):
    """v8: single-pass stopping criterion with two triggers.

    (a) json_complete — the 'decision' field appeared. Since v5 the schema
        places 'decision' as the LAST field, so the JSON object is complete
        and everything after it is waste. Lossless early stop.
    (b) timestamp_loop — abnormal density of HH:MM:SS patterns in the tail
        window: signature of a counting loop (timestamps incrementing one
        second at a time), which repetition_penalty and no_repeat_ngram_size
        structurally CANNOT stop because every step emits novel tokens.

    The decode check runs every STOP_CHECK_EVERY tokens and decodes only
    the last TS_WINDOW_TOKENS tokens to keep overhead negligible.
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer  = tokenizer
        self.prompt_len = prompt_len
        self.loop_detected = False
        self._last_checked_len = 0

    def __call__(self, input_ids, scores, **kwargs):
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len - self._last_checked_len < STOP_CHECK_EVERY:
            return False
        self._last_checked_len = gen_len

        tail_start = max(self.prompt_len, input_ids.shape[1] - TS_WINDOW_TOKENS)
        tail = self.tokenizer.decode(
            input_ids[0][tail_start:], skip_special_tokens=True
        )

        if DECISION_STOP_RE.search(tail):
            return True

        if len(TIMESTAMP_RE.findall(tail)) >= TS_LOOP_THRESHOLD:
            self.loop_detected = True
            return True

        return False


def limpiar_thinking(texto):
    """
    Remueve el bloque <think>...</think> que Qwen3.5 puede emitir antes del JSON.
    CRITICO: sin esto, json.loads() falla porque el texto empieza con <think> y no con '{'.
    v6: also handles the orphan </think> closing tag (the chat template
    injects <think>\\n in the assistant prefix, so the model may emit
    only the closing tag).
    """
    if texto is None:
        return ""
    texto = re.sub(r"<think>.*?</think>\s*", "", texto, flags=re.DOTALL)
    if "</think>" in texto:
        texto = texto.split("</think>", 1)[1]
    return texto.strip()


def extraer_decision(texto):
    match = DECISION_RE.search(texto)
    if match:
        return match.group(1)
    if "APROBADO" in texto.upper():
        return "APROBADO"
    if "RECHAZADO" in texto.upper():
        return "RECHAZADO"
    return "DESCONOCIDO"


def extraer_decision_estricta(texto):
    """v8: field-only extraction for degenerate outputs (truncated / loop).

    The substring fallback of extraer_decision() can fabricate a label from
    'APROBADO'/'RECHAZADO' appearing inside the observations of a half-emitted
    JSON, polluting vqa_ok_modelo and feeding the agreement gate a ghost
    prediction. Degenerate outputs only get a label if the actual field made
    it out.
    """
    match = DECISION_RE.search(texto or "")
    return match.group(1) if match else "DESCONOCIDO"


def extraer_json(texto):
    """
    v6: robust JSON extraction ported from the thinking script.
    Direct parse first; if it fails, regex the outermost {...} block.
    Returns dict or None.
    """
    if texto is None or not texto.strip():
        return None
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        pass
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # v9: brace balancing — recover a JSON whose closing brace(s) never arrived.
    depth, in_str, esc = 0, False, False
    for ch in texto:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if depth > 0:
        candidato = texto.strip() + ('"' if in_str else "") + ("}" * depth)
        try:
            parsed = json.loads(candidato)
            STATS["json_reparados"] += 1
            return parsed
        except json.JSONDecodeError:
            pass
    return None


def calcular_decision_por_reglas(prediccion_texto, label_pred_modelo=None):
    """
    Aplica el arbol de decision sobre el JSON parseado del modelo.

    AGREEMENT GATE (v6): if label_pred_modelo is provided and the rule-based
    decision contradicts the model's own latent prediction, the case is
    escalated instead of auto-decided (motivo 'conflicto_regla_vs_modelo').
    """
    parsed = extraer_json(prediccion_texto)
    if parsed is None:
        return {
            'decision_regla':           'INDETERMINADO',
            'confidence_real':          0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple':          None,
            'decision_final':           'REQUIERE_REVISION_MANUAL',
            'motivo':                   'json_invalido_o_truncado',
            'acreditacion':             {},
            'punto':                    '',
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
            'acreditacion':             {},
            'punto':                    '',
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

    # ── v10: CAPS DE ACREDITACION ────────────────────────────
    # Mirror of the confidence caps declared in prompt v10. The model REPORTS
    # observables in `acreditacion`; the engine decides. Capping to 0.60 (below
    # THRESHOLD_REVISION) makes the existing threshold escalate the case, so no
    # separate escalation path is needed. Backward compatible: a v9 prediction
    # has no `acreditacion` block and no cap fires.
    #
    # Escalate, do NOT auto-reject: the reject arm measured 87% precision, but on
    # 15 cases and with signals chosen while looking at those same cases. Wrongly
    # rejecting a distributor's claim is the costliest error in this application.
    acr   = parsed.get('acreditacion') or {}
    punto = str(parsed.get('punto_arbol_aplicado', ''))
    caps  = []
    if acr.get('causa_fisica_fotografiada') == 'no':
        caps.append('causa_no_acreditada')
    if acr.get('prueba_no_prevenibilidad') == 'ninguna' and punto in ('1', '3'):
        caps.append('sin_prueba_no_prevenibilidad')
    if acr.get('solicitante_identificado') == 'indeterminado':
        caps.append('solicitante_indeterminado')
    # v10.1: coherence rule. "no_aplica" is defined ONLY for descargos (7a/7b/8)
    # and transmission/AMM events (4/5/6). On a field failure it is incoherent
    # by definition — and it is exactly the escape the train-split run showed:
    # the model NEVER emitted "no" (which trips the cap) and routed the cases
    # that warranted it through "no_aplica" (which did not). Close the gap.
    if punto in ('1', '2', '3', '9') and acr.get('causa_fisica_fotografiada') == 'no_aplica':
        caps.append('acreditacion_incoherente_causa')
    # Only punto 3: it is BY DEFINITION an object near the grid, so the
    # non-preventability field cannot be "no_aplica" there. Punto 1 is excluded
    # on purpose: lightning / landslide cases legitimately carry "no_aplica".
    if punto == '3' and acr.get('prueba_no_prevenibilidad') == 'no_aplica':
        caps.append('acreditacion_incoherente_prueba')
    if caps and confidence_real > CAP_ACREDITACION:
        confidence_real = CAP_ACREDITACION
        motivo = f"{motivo}_cap_{'+'.join(caps)}"

    if decision_regla == 'INDETERMINADO':
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final   = motivo
    elif confidence_real < THRESHOLD_REVISION:
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final   = f'{motivo}_confidence_bajo'
    else:
        decision_final = decision_regla
        motivo_final   = motivo

    # ── AGREEMENT GATE (v6) ──────────────────────────────────
    # Never auto-decide against the model's own latent prediction.
    if (decision_final in ('APROBADO', 'RECHAZADO')
            and label_pred_modelo in ('APROBADO', 'RECHAZADO')
            and decision_final != label_pred_modelo):
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final   = f'{motivo}_conflicto_regla_vs_modelo'

    return {
        'decision_regla':           decision_regla,
        'confidence_real':          round(confidence_real, 2),
        'criterios_cumplidos_real': criterios_cumplidos_real,
        'causa_fm_cumple':          causa_fm_cumple,
        'decision_final':           decision_final,
        'motivo':                   motivo_final,
        'acreditacion':             acr,
        'punto':                    punto,
    }


def analizar_prediccion(prediccion, truncado, loop_abortado):
    """v8: single entry point that applies the direct-escalation policy.

    Degenerate outputs (loop aborted or truncated) escalate immediately:
    no rules, no agreement gate, strict label extraction only.
    Returns (label_pred_modelo, analisis_dict).
    """
    if loop_abortado or truncado:
        label_pred_modelo = extraer_decision_estricta(prediccion)
        motivo = 'loop_ciclico_abortado' if loop_abortado else 'salida_truncada'
        return label_pred_modelo, {
            'decision_regla':           'INDETERMINADO',
            'confidence_real':          0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple':          None,
            'decision_final':           'REQUIERE_REVISION_MANUAL',
            'motivo':                   motivo,
            'acreditacion':             {},
            'punto':                    '',
        }

    label_pred_modelo = extraer_decision(prediccion)
    analisis = calcular_decision_por_reglas(prediccion, label_pred_modelo)
    return label_pred_modelo, analisis


# ══════════════════════════════════════════
# 4b. Re-procesar checkpoint con parser + escalacion directa v8,
#     y expulsar SOLO los casos re-inferibles.
#     (Va aqui porque las funciones se definen despues del bloque 3.)
# ══════════════════════════════════════════
if resultados:
    n_regateados = 0
    for r in resultados:
        if not r.get("prediccion"):
            continue
        prediccion_limpia = limpiar_thinking(r["prediccion"])
        label_real        = r["label_real"]
        truncado_prev     = bool(r.get("truncado"))
        loop_prev         = bool(r.get("loop_abortado"))

        label_pred_modelo, analisis = analizar_prediccion(
            prediccion_limpia, truncado_prev, loop_prev
        )
        decision_final = analisis['decision_final']

        if decision_final == 'REQUIERE_REVISION_MANUAL':
            es_correcto_auto = None
        else:
            es_correcto_auto = (decision_final == label_real)

        if decision_final != r.get("decision_final"):
            n_regateados += 1
            print(f"  >>> Regateado {r['caso']}: "
                  f"{r.get('decision_final')} -> {decision_final} "
                  f"({analisis['motivo']})")

        r["prediccion"]               = prediccion_limpia
        r["label_pred_modelo"]        = label_pred_modelo
        r["vqa_ok_modelo"]            = (label_pred_modelo == label_real)
        r["decision_regla"]           = analisis['decision_regla']
        r["decision_final"]           = decision_final
        r["confidence_real"]          = analisis['confidence_real']
        r["criterios_cumplidos_real"] = analisis['criterios_cumplidos_real']
        r["causa_fm_cumple"]          = analisis['causa_fm_cumple']
        r["motivo"]                   = analisis['motivo']
        r["es_correcto_auto"]         = es_correcto_auto
        r["loop_abortado"]            = loop_prev

    def _necesita_reinferencia(r):
        # v8 BUG FIX: greedy decoding is deterministic — a case that ended in
        # a counting loop reproduces the SAME loop on every checkpoint resume.
        # Cases already flagged loop_abortado are FINAL (escalated), never
        # re-inferred. Legacy broken cases (old checkpoints without the flag)
        # get exactly one retry, now protected by the loop guard.
        if r.get("loop_abortado"):
            return False
        return (r.get("truncado")
                or r.get("label_pred_modelo") == "DESCONOCIDO"
                or r.get("motivo") in ("json_invalido_o_truncado", "sin_criterios"))

    a_reinferir = [r["caso"] for r in resultados if _necesita_reinferencia(r)]
    if a_reinferir:
        print(f"\n>>> {len(a_reinferir)} casos rotos (legacy) se re-infieren "
              f"con loop guard: {a_reinferir}")
        resultados = [r for r in resultados if not _necesita_reinferencia(r)]

    if n_regateados > 0 or a_reinferir:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False)
        print(f">>> Checkpoint actualizado: {n_regateados} decisiones "
              f"recalculadas con politica v8.")

    casos_procesados = {r["caso"] for r in resultados}
    print(f"Casos pendientes de inferencia: {n_total - len(resultados)}")


def inferir_caso(caso):
    """Ejecuta inferencia. Retorna (texto_limpio, n_tokens, truncado, loop_abortado)."""
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
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,   # <-- Qwen3.5 specific
    )
    images = [Image.open(ruta).convert("RGB") for ruta in rutas]

    inputs = tokenizer(
        text=text_input,
        images=images,
        return_tensors="pt",
        truncation=False,
    ).to("cuda")

    n_input_tokens = inputs["input_ids"].shape[1]
    stopper        = JsonCompleteOrLoopStop(tokenizer, n_input_tokens)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=REPETITION_PENALTY,
            # NOTA v7: no_repeat_ngram_size fue probado y REVERTIDO. Rompe el
            # boilerplate legitimo de los 7 bloques de criterios ("cumple":
            # true, "observacion": "...") y el parser deja de poder extraer
            # 'criterios' en el 100% de los casos.
            # NOTA v8: los loops de conteo se cortan via StoppingCriteria,
            # no via decoding params — ver JsonCompleteOrLoopStop.
            stopping_criteria=StoppingCriteriaList([stopper]),
            pad_token_id=tokenizer.eos_token_id,
        )

    n_generated   = output_ids.shape[1] - n_input_tokens
    last_token    = output_ids[0, -1].item()
    loop_abortado = stopper.loop_detected
    truncado      = (not loop_abortado
                     and last_token != tokenizer.eos_token_id
                     and n_generated >= MAX_NEW_TOKENS)

    generated_raw = tokenizer.decode(
        output_ids[0][n_input_tokens:],
        skip_special_tokens=True
    )

    # CRITICO: limpiar <think> antes de devolver, para que json.loads funcione
    generated = limpiar_thinking(generated_raw)

    # Liberar tensores y cerrar imagenes
    del inputs, output_ids
    torch.cuda.empty_cache()
    for img in images:
        img.close()

    return generated, n_generated, truncado, loop_abortado


# ══════════════════════════════════════════
# 5. Loop de inferencia
# ══════════════════════════════════════════
tiempos_casos = [r["tiempo_s"] for r in resultados]
total_inicio  = time.time()
casos_nuevos  = 0

print(f"\nInferencia sobre {n_total - len(casos_procesados)} casos pendientes.\n")

for i, caso in enumerate(casos):

    if caso["id"] in casos_procesados:
        print(f"  [{i+1:3d}/{n_total}] {caso['id']} — skipped (checkpoint)")
        continue

    caso_inicio = time.time()
    label_real  = caso["label_real"]

    print(f"  [{i+1:3d}/{n_total}] {caso['id']}...", end='', flush=True)

    prediccion, n_tokens, truncado, loop_abortado = inferir_caso(caso)
    tiempo_caso = time.time() - caso_inicio

    # v8: direct escalation for degenerate outputs, strict label extraction
    label_pred_modelo, analisis = analizar_prediccion(
        prediccion, truncado, loop_abortado
    )
    vqa_ok_modelo  = (label_pred_modelo == label_real)
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

    casos_restantes = n_total - (i + 1)
    avg_tiempo      = sum(tiempos_casos) / len(tiempos_casos)
    eta_min         = casos_restantes * avg_tiempo / 60

    tag = " LOOP" if loop_abortado else (" TRUNC" if truncado else "")
    print(f" Real:{label_real[:4]:<4} Modelo:{label_pred_modelo[:4]:<4} "
          f"Final:{decision_final[:4]:<4} C:{analisis['criterios_cumplidos_real']}/7 "
          f"Conf:{analisis['confidence_real']:.2f} {estado}{tag} "
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
        "loop_abortado":            loop_abortado,
        "acreditacion":             analisis.get('acreditacion', {}),
        "punto":                    analisis.get('punto', ''),
        "tiempo_s":                 round(tiempo_caso, 2),
    })

    if casos_nuevos % CHECKPOINT_INTERVAL == 0:
        with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False)
        print(f"  >>> Checkpoint guardado ({len(resultados)}/{n_total})")

tiempo_total = time.time() - total_inicio

# ══════════════════════════════════════════
# 6. Metricas — por split train / test / total
# ══════════════════════════════════════════
# IMPORTANT: checkpoint recovery + re-inference can append re-run cases at
# the END of `resultados`, so slicing resultados[:100] would mix splits.
# We map each case back to its index in the dataset instead.
orden_dataset = {c["id"]: i for i, c in enumerate(casos)}
resultados.sort(key=lambda r: orden_dataset.get(r["caso"], 10**9))

N_TRAIN = 100   # first 100 dataset cases were seen in training; last 100 are unseen

resultados_train = [r for r in resultados
                    if orden_dataset.get(r["caso"], 10**9) < N_TRAIN]
resultados_test  = [r for r in resultados
                    if orden_dataset.get(r["caso"], 10**9) >= N_TRAIN]
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


def metricas_por_split(subset):
    """Total + per-class metrics for one split (train, test or full)."""
    return {
        "total":      calcular_metricas(subset),
        "aprobados":  calcular_metricas([r for r in subset if r["label_real"] == "APROBADO"]),
        "rechazados": calcular_metricas([r for r in subset if r["label_real"] == "RECHAZADO"]),
    }


m_train = metricas_por_split(resultados_train)
m_test  = metricas_por_split(resultados_test)
m_full  = metricas_por_split(resultados)

# Backward-compatible aliases (rest of the script and output JSON use these)
m     = m_full["total"]
m_apr = m_full["aprobados"]
m_rch = m_full["rechazados"]

# ══════════════════════════════════════════
# 7. Imprimir resultados
# ══════════════════════════════════════════
def imprimir_metricas(m, titulo):
    if m is None:
        print(f"\n  [{titulo}] split vacio — sin metricas.")
        return
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


def imprimir_split(ms, titulo):
    print(f"\n{'#'*70}")
    print(f"##  {titulo}")
    print(f"{'#'*70}")
    imprimir_metricas(ms["total"],      f"{titulo} — TOTAL")
    imprimir_metricas(ms["aprobados"],  f"{titulo} — APROBADOS (ground truth)")
    imprimir_metricas(ms["rechazados"], f"{titulo} — RECHAZADOS (ground truth)")


imprimir_split(m_train, f"TRAIN — Primeros {N_TRAIN} casos (vistos en training)")
imprimir_split(m_test,  f"TEST — Ultimos {len(resultados_test)} casos (nunca vistos)")
imprimir_split(m_full,  "TOTAL — Todos los casos")

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

print(f"\n{'='*70}")
print(f"  CASOS DEGENERADOS (loop / truncado) — escalados directo")
print(f"{'='*70}")
degenerados = [r for r in resultados if r.get("loop_abortado") or r.get("truncado")]
print(f"  Total degenerados: {len(degenerados)}/{n_total}")
for r in degenerados:
    tipo = "LOOP" if r.get("loop_abortado") else "TRUNC"
    print(f"    {r['caso']}: [{tipo}] real={r['label_real']:<10} "
          f"tokens={r['tokens_generados']}  motivo={r['motivo']}")

# ══════════════════════════════════════════
# 9. Tiempos y tokens
# ══════════════════════════════════════════
tokens_lista     = [r["tokens_generados"] for r in resultados]
n_truncados      = sum(1 for r in resultados if r["truncado"])
n_loops          = sum(1 for r in resultados if r.get("loop_abortado"))

print(f"\n{'='*70}")
print(f"  TIEMPOS Y TOKENS")
print(f"{'='*70}")
print(f"  Tiempo total sesion: {tiempo_total:.0f}s ({tiempo_total/60:.1f} min)")
print(f"  Promedio/caso:       {sum(tiempos_casos)/len(tiempos_casos):.1f}s")
print(f"  Tokens: min={min(tokens_lista)} media={int(np.mean(tokens_lista))} "
      f"P50={int(np.median(tokens_lista))} P95={int(np.percentile(tokens_lista,95))} "
      f"max={max(tokens_lista)}")
print(f"  Truncados: {n_truncados}/{n_total} ({n_truncados/n_total*100:.1f}%)")
print(f"  Loops abortados: {n_loops}/{n_total} ({n_loops/n_total*100:.1f}%)")
print(f"  JSON reparados por balanceo de llaves: {STATS['json_reparados']}")

# ══════════════════════════════════════════
# 10. Guardar JSON final y limpiar checkpoint
# ══════════════════════════════════════════
output = {
    "config": {
        "max_imagenes":           MAX_IMAGENES,
        "max_new_tokens":         MAX_NEW_TOKENS,
        "max_seq_len":            MAX_SEQ_LEN,
        "threshold_revision":     THRESHOLD_REVISION,
        "no_repeat_ngram_size":   None,  # v7: reverted — breaks the repeated criterios JSON boilerplate
        "repetition_penalty":     REPETITION_PENALTY,
        "agreement_gate":         True,                   # v6: rule/model conflict -> escalation
        "stop_on_decision":       "requires_closing_brace",  # v9: never halts on an unclosed JSON object
        "json_reparados":         STATS["json_reparados"],    # v9: brace-repair hits
        "loop_guard":             {                       # v8: counting-loop abort -> direct escalation
            "check_every":    STOP_CHECK_EVERY,
            "window_tokens":  TS_WINDOW_TOKENS,
            "ts_threshold":   TS_LOOP_THRESHOLD,
        },
        "enable_thinking":        ENABLE_THINKING,
        "model_path":             MODEL_PATH,
        "dataset_path":           DATASET_PATH,
    },
    "n_total":            n_total,
    "n_train":            len(resultados_train),
    "n_test":             len(resultados_test),
    "metricas_total":     m,
    "metricas_aprobados": m_apr,
    "metricas_rechazados":m_rch,
    "metricas_train":     m_train,    # total/aprobados/rechazados of first N_TRAIN cases
    "metricas_test":      m_test,     # total/aprobados/rechazados of unseen cases
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
        "n_loops":     int(n_loops),
    },
    "resultados": resultados,
}

output_path = f"{OUTPUT_DIR}/resultados_inferencia.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nResultados guardados en: {output_path}")

if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint eliminado (run completo).")

# ══════════════════════════════════════════
# 11. Graficas — una por split (train / test / total)
# ══════════════════════════════════════════
def generar_grafica(ms, titulo, path):
    """4-panel figure for one split: accuracy per class, decision
    distribution, system confusion matrix and global indicators."""
    mt, ma, mr = ms["total"], ms["aprobados"], ms["rechazados"]
    if mt is None or ma is None or mr is None:
        print(f"  [{titulo}] split vacio o sin ambas clases — grafica omitida.")
        return
    n_split = mt["n"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"{titulo} (n={n_split}) — Qwen3.5 9B (CNEE)",
                 fontsize=13, fontweight="bold")

    # Plot 1: Accuracy modelo vs sistema por clase
    ax = axes[0, 0]
    labels   = ['APROBADOS\n(modelo)', 'APROBADOS\n(sistema)', 'RECHAZADOS\n(modelo)', 'RECHAZADOS\n(sistema)']
    vals_acc = [
        ma['vqa_pct'],
        ma['tasa_auto_segura'],
        mr['vqa_pct'],
        mr['tasa_auto_segura'],
    ]
    vals_fp = [
        ma['confusion_m']['fp'] / ma['n'] * 100,
        ma['tasa_falsos_auto'],
        mr['confusion_m']['fp'] / mr['n'] * 100,
        mr['tasa_falsos_auto'],
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
    vals = [mt['n_aprobado'], mt['n_rechazado'], mt['n_escalado']]
    cols = ['#16A34A', '#0D1B3E', '#F59E0B']
    bars = ax.bar(cats, vals, color=cols, width=0.6)
    ax.set_title(f"Distribucion de Decisiones Finales (n={n_split})")
    ax.set_ylabel("Cantidad de casos")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals):
        pct = v / n_split * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v}\n({pct:.0f}%)", ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    # Plot 3: Confusion matrix sistema
    ax     = axes[1, 0]
    n_auto = n_split - mt['n_escalado']
    confusion = [[mt['confusion_s']['tp'], mt['confusion_s']['fn']],
                 [mt['confusion_s']['fp'], mt['confusion_s']['tn']]]
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred APROBADO", "Pred RECHAZADO"])
    ax.set_yticklabels(["Real APROBADO", "Real RECHAZADO"])
    ax.set_title(f"Matriz de Confusion — Sistema (auto, n={n_auto})")
    for i in range(2):
        for j in range(2):
            denom = max(n_auto, 1)
            ax.text(j, i, str(confusion[i][j]),
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color="white" if confusion[i][j] > denom / 4 else "black")

    # Plot 4: Indicadores globales
    ax          = axes[1, 1]
    indicadores = ['VQA\nAccuracy', 'Auto\nSegura', 'Falsos+\nAuto', 'Escalacion']
    vals_bar    = [mt['vqa_pct'], mt['tasa_auto_segura'], mt['tasa_falsos_auto'], mt['tasa_escalacion']]
    cols_bar    = ['#16A34A', '#16A34A', '#DC2626', '#F59E0B']
    bars        = ax.bar(indicadores, vals_bar, color=cols_bar, width=0.5)
    ax.set_title("Indicadores Globales")
    ax.set_ylabel("Porcentaje (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, vals_bar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{v:.1f}%",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Grafica guardada en: {path}")


generar_grafica(m_train,
                f"Train — Primeros {N_TRAIN} casos (vistos)",
                f"{OUTPUT_DIR}/inferencia_resultados_train.png")
generar_grafica(m_test,
                f"Test — Ultimos {len(resultados_test)} casos (nunca vistos)",
                f"{OUTPUT_DIR}/inferencia_resultados_test.png")
generar_grafica(m_full,
                "Total — Todos los casos",
                f"{OUTPUT_DIR}/inferencia_resultados_total.png")

print("\n=== Inferencia completada ===")