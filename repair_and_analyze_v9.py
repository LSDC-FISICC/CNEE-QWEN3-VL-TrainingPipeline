"""
Reparacion offline y re-analisis de resultados_inferencia.json (run v8, prompt v9).

Aborda dos problemas independientes:

(A) LLAVE DE CIERRE FALTANTE — el criterio stop-on-decision disparaba en cuanto
    aparecia '"decision": "APROBADO|RECHAZADO"' en la ventana decodificada, que
    se revisa cada STOP_CHECK_EVERY=8 tokens. Cuando la llave de cierre '}' aun
    no se habia emitido en ese chequeo, la generacion paraba un token antes y el
    JSON quedaba inparseable -> 'json_invalido_o_truncado' -> escalacion forzada.
    El campo 'prediccion' quedo INTACTO en el checkpoint, asi que esto se repara
    offline balanceando las llaves y re-parseando. NO requiere re-inferencia.

(B) CONCENTRACION EN PUNTO 1 — gate opcional que enruta todo caso climatico
    (punto_arbol_aplicado == "1") a revision manual. Se simula para cuantificar
    el trade-off antes de decidir si se adopta.

Uso:
    python reparar_y_reanalizar_v9.py resultados_inferencia.json
"""

import glob
import json
import os
import re
import sys
from collections import Counter

THRESHOLD_REVISION = 0.75
N_TRAIN = 100

DECISION_RE = re.compile(r'"decision"\s*:\s*"(APROBADO|RECHAZADO)"')


# ══════════════════════════════════════════
# 1. Reparacion de JSON truncado
# ══════════════════════════════════════════
def _analizar_llaves(t):
    """Return (depth, inside_string) ignoring braces that live inside strings."""
    depth = 0
    in_str = False
    esc = False
    for ch in t:
        if esc:
            esc = False
            continue
        if ch == '\\':
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
    return depth, in_str


def reparar_json(texto):
    """Parse JSON, repairing an unterminated tail when possible.

    Returns (dict_or_None, was_repaired).
    """
    if not texto or not texto.strip():
        return None, False
    t = texto.strip()

    # (1) direct parse
    try:
        return json.loads(t), False
    except (json.JSONDecodeError, TypeError):
        pass

    # (2) outermost {...} block
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), False
        except json.JSONDecodeError:
            pass

    # (3) brace balancing: close the open string (if any) and append '}'
    depth, in_str = _analizar_llaves(t)
    if depth > 0:
        candidato = t + ('"' if in_str else '') + ('}' * depth)
        try:
            return json.loads(candidato), True
        except json.JSONDecodeError:
            pass
        # drop a trailing incomplete key/value pair, then retry
        corte = max(t.rfind('},'), t.rfind('",'))
        if corte > 0:
            recorte = t[:corte + 1]
            d2, s2 = _analizar_llaves(recorte)
            if d2 > 0:
                try:
                    return json.loads(recorte + ('}' * d2)), True
                except json.JSONDecodeError:
                    pass
    return None, False


# ══════════════════════════════════════════
# 2. Reglas de decision (identicas al harness)
# ══════════════════════════════════════════
def decidir(parsed, label_pred_modelo):
    if parsed is None:
        return {'decision_final': 'REQUIERE_REVISION_MANUAL',
                'motivo': 'json_irrecuperable',
                'criterios_cumplidos_real': 0, 'confidence_real': 0.0,
                'punto': None}

    criterios = parsed.get('criterios', {})
    punto = str(parsed.get('punto_arbol_aplicado', '')) or None
    if not criterios:
        return {'decision_final': 'REQUIERE_REVISION_MANUAL',
                'motivo': 'sin_criterios',
                'criterios_cumplidos_real': 0, 'confidence_real': 0.0,
                'punto': punto}

    n_ok = sum(1 for c in criterios.values() if c.get('cumple') is True)
    causa = criterios.get('causa_fuerza_mayor', {}).get('cumple')

    if causa is False:
        regla, conf, motivo = 'RECHAZADO', 0.90, 'causa_FM_no_cumple'
    elif causa is True and n_ok >= 5:
        regla = 'APROBADO'
        if n_ok == 7:
            conf, motivo = 0.95, 'aprobado_7de7'
        elif n_ok == 6:
            conf, motivo = 0.82, 'aprobado_6de7'
        else:
            conf, motivo = 0.65, 'aprobado_5de7_zona_gris'
    elif causa is True and n_ok < 5:
        regla, conf, motivo = 'RECHAZADO', 0.55, 'rechazado_docs_insuficientes'
    else:
        regla, conf, motivo = 'INDETERMINADO', 0.0, 'causa_no_determinable'

    if regla == 'INDETERMINADO':
        final, motivo_f = 'REQUIERE_REVISION_MANUAL', motivo
    elif conf < THRESHOLD_REVISION:
        final, motivo_f = 'REQUIERE_REVISION_MANUAL', f'{motivo}_confidence_bajo'
    else:
        final, motivo_f = regla, motivo

    # agreement gate
    if (final in ('APROBADO', 'RECHAZADO')
            and label_pred_modelo in ('APROBADO', 'RECHAZADO')
            and final != label_pred_modelo):
        final, motivo_f = 'REQUIERE_REVISION_MANUAL', f'{motivo}_conflicto_regla_vs_modelo'

    return {'decision_final': final, 'motivo': motivo_f,
            'criterios_cumplidos_real': n_ok, 'confidence_real': conf,
            'punto': punto}


# ══════════════════════════════════════════
# 3. Metricas
# ══════════════════════════════════════════
def metricas(subset):
    n = len(subset)
    if n == 0:
        return None
    vqa = sum(1 for r in subset if r['label_pred_modelo'] == r['label_real'])
    tp = sum(1 for r in subset if r['label_real'] == 'APROBADO' and r['decision_final'] == 'APROBADO')
    tn = sum(1 for r in subset if r['label_real'] == 'RECHAZADO' and r['decision_final'] == 'RECHAZADO')
    fp = sum(1 for r in subset if r['label_real'] == 'RECHAZADO' and r['decision_final'] == 'APROBADO')
    fn = sum(1 for r in subset if r['label_real'] == 'APROBADO' and r['decision_final'] == 'RECHAZADO')
    esc = sum(1 for r in subset if r['decision_final'] == 'REQUIERE_REVISION_MANUAL')
    return {'n': n, 'vqa': vqa, 'vqa_pct': vqa / n * 100,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'esc': esc,
            'auto_ok': (tp + tn) / n * 100,
            'auto_fail': (fp + fn) / n * 100,
            'escalacion': esc / n * 100}


def imprimir(m, titulo):
    if m is None:
        print(f"  [{titulo}] vacio")
        return
    print(f"  {titulo:<34} n={m['n']:>3} | VQA {m['vqa_pct']:5.1f}% | "
          f"auto-ok {m['auto_ok']:5.1f}% | auto-fail {m['auto_fail']:5.1f}% | "
          f"escal {m['escalacion']:5.1f}% | TP{m['tp']:>3} TN{m['tn']:>3} FP{m['fp']:>3} FN{m['fn']:>3}")


def parsear_estricto(texto):
    """Original harness behaviour: direct parse, then outermost {...}. No repair."""
    if not texto or not texto.strip():
        return None, False
    try:
        return json.loads(texto.strip()), False
    except (json.JSONDecodeError, TypeError):
        pass
    m = re.search(r"\{.*\}", texto, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), False
        except json.JSONDecodeError:
            pass
    return None, False


def evaluar(resultados, gate_punto1=False, reparar=True):
    out = []
    for r in resultados:
        parsed, reparado = (reparar_json(r.get('prediccion', '')) if reparar
                            else parsear_estricto(r.get('prediccion', '')))
        lab = r['label_pred_modelo']
        d = decidir(parsed, lab)
        if gate_punto1 and d['punto'] == '1' and d['decision_final'] != 'REQUIERE_REVISION_MANUAL':
            d['decision_final'] = 'REQUIERE_REVISION_MANUAL'
            d['motivo'] = 'gate_clima_punto1'
        out.append({**r, **d, 'reparado': reparado})
    return out


def reporte(resultados, titulo, gate_punto1=False, reparar=True):
    ev = evaluar(resultados, gate_punto1, reparar)
    train, test = ev[:N_TRAIN], ev[N_TRAIN:]
    print(f"\n{'='*118}\n{titulo}\n{'='*118}")
    for nombre, sub in (('TOTAL', ev), ('TRAIN (vistos)', train), ('TEST (no vistos)', test)):
        imprimir(metricas(sub), f'{nombre} — todos')
        imprimir(metricas([r for r in sub if r['label_real'] == 'APROBADO']), f'{nombre} — aprobados')
        imprimir(metricas([r for r in sub if r['label_real'] == 'RECHAZADO']), f'{nombre} — rechazados')
        print()
    return ev


def localizar_resultados(path=None):
    """Resolve the results file: explicit arg, else search under ./output."""
    if path and os.path.isfile(path):
        return path
    if path:
        print(f"No existe: {path}\nBuscando alternativas...")
    crudos = (glob.glob('output/**/resultados_inferencia.json', recursive=True) +
              glob.glob('**/resultados_inferencia.json', recursive=True) +
              glob.glob('output/**/checkpoint_inferencia.json', recursive=True))
    vistos = set()
    candidatos = []
    for c in crudos:                      # dedupe by real path, keep first sighting
        real = os.path.realpath(c)
        if real not in vistos:
            vistos.add(real)
            candidatos.append(c)
    candidatos.sort(key=os.path.getmtime, reverse=True)
    if not candidatos:
        print("No se encontro ningun resultados_inferencia.json ni checkpoint_inferencia.json.")
        print("Pasa la ruta explicitamente:  python reparar_y_reanalizar_v9.py <ruta>")
        sys.exit(1)
    if len(candidatos) > 1:
        print("Archivos encontrados (se usa el mas reciente):")
        for c in candidatos:
            print(f"   {c}")
    print(f"Usando: {candidatos[0]}\n")
    return candidatos[0]


def cargar(path):
    """Accept both the final output file and a raw checkpoint array."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):                 # checkpoint_inferencia.json
        return {'resultados': data}, data
    return data, data['resultados']


def main(path=None):
    path = localizar_resultados(path)
    data, resultados = cargar(path)
    print(f"Casos cargados: {len(resultados)}")

    # --- diagnostico de la reparacion ---
    n_rep = 0
    n_irrec = 0
    for r in resultados:
        parsed, rep = reparar_json(r.get('prediccion', ''))
        if rep:
            n_rep += 1
        if parsed is None:
            n_irrec += 1
    print(f"JSON reparados por balanceo de llaves: {n_rep}")
    print(f"JSON irrecuperables (truncamiento real): {n_irrec}")

    base = reporte(resultados, "A) ESTADO ORIGINAL (sin reparar) — replica del run",
                   reparar=False)
    rep = reporte(resultados, "B) CON JSON REPARADO")
    gate = reporte(resultados, "C) CON JSON REPARADO + GATE CLIMA (punto 1 -> revision manual)",
                   gate_punto1=True)

    # --- concentracion de falsos positivos por punto del arbol ---
    print(f"{'='*118}\nFALSOS POSITIVOS DEL MODELO POR PUNTO DEL ARBOL (test)\n{'='*118}")
    fp_puntos = Counter()
    for r in rep[N_TRAIN:]:
        if r['label_real'] == 'RECHAZADO' and r['label_pred_modelo'] == 'APROBADO':
            fp_puntos[r['punto']] += 1
    total_fp = sum(fp_puntos.values())
    for punto, cnt in sorted(fp_puntos.items(), key=lambda x: -x[1]):
        print(f"  punto {str(punto):<3} : {cnt:>3}  ({cnt/max(total_fp,1)*100:.0f}% de los FP)")
    print(f"  TOTAL FP modelo (test): {total_fp}")

    # --- guardar version reparada ---
    salida = path.replace('.json', '_reparado.json')
    data['resultados'] = rep
    with open(salida, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\nResultados reparados guardados en: {salida}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)