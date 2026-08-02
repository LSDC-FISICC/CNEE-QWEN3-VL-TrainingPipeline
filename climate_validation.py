"""
Validacion de hipotesis sobre el discriminador real de los casos de clima.

HIPOTESIS (derivada de leer los expedientes del run v9, requiere confirmacion):
  Los casos climaticos APROBADOS por DRCS se distinguen de los RECHAZADOS por
  DOS artefactos verificables en el expediente, no por el boletin meteorologico:

    H1. FOTO DE LA CAUSA FISICA en el punto de falla (arbol, rama, aislador,
        pararrayo, poste quebrado, objeto). Si la unica evidencia fotografica
        es el DISPOSITIVO DE PROTECCION (fusible quemado, TripSaver abierto,
        reconectador aperturado) o la maniobra, la causa no esta acreditada.

    H2. PRUEBA DE NO-PREVENIBILIDAD verificable: medicion de distancia entre
        la base de la vegetacion y la red (vs. la franja de 1.4 m), oficio
        CONAP/INAB, carta del propietario negando la poda, o evidencia de
        intervencion de terceros sobre la vegetacion.

Este script clasifica cada caso segun H1/H2 leyendo las observaciones del JSON
predicho y cruza contra label_real. Si la hipotesis es correcta, la tabla debe
mostrar separacion fuerte en los casos de punto 1.

NOTA: opera sobre el texto que el MODELO escribio describiendo el expediente,
que puede omitir o inventar detalles. Sirve para priorizar la revision manual,
no como verdad. Confirmar sobre una muestra antes de rehacer el dataset.

Uso:
    python validar_hipotesis_clima.py resultados_inferencia.json
"""

import json
import re
import sys
from collections import defaultdict

# ── H1: causa fisica visible vs. solo dispositivo/maniobra ───────────
CAUSA_FISICA = re.compile(
    r'(árbol|arbol|rama|palma|bambú|bambu|banano|vegetaci|musgo|nido|'
    r'ave\b|animal|ardilla|perro|gato|'
    r'aislador|pararrayo|crucero|retenida|ancla|'
    r'poste\s+(de\s+\w+\s+)?(quebrad|dañad|desplomad|volcad|caíd|caid|en el suelo)|'
    r'vehículo|vehiculo|camión|camion|choque|impacto|'
    r'nylon|rótulo|rotulo|rotulo|objeto extraño|objeto extrano|fibra óptica|fibra optica|'
    r'línea.{0,20}(rota|reventada|quemada)|linea.{0,20}(rota|reventada|quemada)|'
    r'deslave|roca|incendio|quema de (caña|cana|cohetillo|plantación|plantacion))',
    re.IGNORECASE)

SOLO_DISPOSITIVO = re.compile(
    r'(fusible\s+(quemado|activad|en modo)|protección tipo fusible|proteccion tipo fusible|'
    r'tripsaver|trip saver|trip saber|'
    r'reconectador\s+(aperturad|de cabecera|de media)|recloser|'
    r'seccionamiento|cuchillas|breaker|interruptor bloqueado|'
    r'elemento maniobrado|modo desconexión|modo desconexion|triple disparo)',
    re.IGNORECASE)

# ── H2: prueba verificable de no-prevenibilidad ──────────────────────
MEDICION = re.compile(
    r'(medición|medicion|distancia).{0,60}?\d+([.,]\d+)?\s*(m\b|mts|metros)|'
    r'\d+([.,]\d+)?\s*(m\b|mts|metros).{0,40}?(de la red|de las líneas|de las lineas|'
    r'base del árbol|base del arbol|librea|libranza|servidumbre)|'
    r'1[.,]4\d?\s*(m\b|mts|metros)', re.IGNORECASE)

TERCERO_DOCUMENTADO = re.compile(
    r'(CONAP|INAB|área protegida|area protegida|zona de vida|'
    r'carta.{0,40}(propietario|rechaz)|rechazando la (solicitud de )?poda|'
    r'talad[oa] por terceras|cortes? (realizados? )?por terceras|poda(do)? por terceras|'
    r'motosierra|corte limpio|tala intencional|'
    r'eliminación de palma|eliminacion de palma|lisofa)', re.IGNORECASE)


def texto_evidencia(parsed):
    """Concatenate the observation fields where physical evidence is described."""
    c = parsed.get('criterios', {})
    campos = ['evidencia_fotografica', 'responsabilidad_externa',
              'causa_fuerza_mayor', 'documentacion_soporte']
    partes = [c.get(k, {}).get('observacion', '') or '' for k in campos]
    partes.append(parsed.get('subcausa') or '')
    partes.append(parsed.get('resumen') or '')
    return ' '.join(partes)


def reparar_json(texto):
    if not texto or not texto.strip():
        return None
    t = texto.strip()
    for cand in (t, ):
        try:
            return json.loads(cand)
        except Exception:
            pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    depth, in_str, esc = 0, False, False
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
    if depth > 0:
        try:
            return json.loads(t + ('"' if in_str else '') + '}' * depth)
        except Exception:
            return None
    return None


def clasificar(parsed):
    txt = texto_evidencia(parsed)
    causa = bool(CAUSA_FISICA.search(txt))
    disp = bool(SOLO_DISPOSITIVO.search(txt))
    return {
        'h1_causa_fisica': causa,
        'h1_solo_dispositivo': (disp and not causa),
        'h2_medicion': bool(MEDICION.search(txt)),
        'h2_tercero': bool(TERCERO_DOCUMENTADO.search(txt)),
    }


def main(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    resultados = data['resultados']

    filas = []
    for r in resultados:
        parsed = reparar_json(r.get('prediccion', ''))
        if parsed is None:
            continue
        cls = clasificar(parsed)
        filas.append({
            'caso': r['caso'],
            'real': r['label_real'],
            'pred': r['label_pred_modelo'],
            'punto': str(parsed.get('punto_arbol_aplicado', '')),
            **cls,
        })

    print(f"Casos analizables: {len(filas)}/{len(resultados)}\n")

    # ── H1 sobre casos de punto 1 (clima) ────────────────────────────
    clima = [f for f in filas if f['punto'] == '1']
    print("=" * 78)
    print(f"H1 — CAUSA FISICA FOTOGRAFIADA  (casos punto 1, n={len(clima)})")
    print("=" * 78)
    tabla = defaultdict(lambda: {'APROBADO': 0, 'RECHAZADO': 0})
    for f in clima:
        key = 'solo dispositivo/maniobra' if f['h1_solo_dispositivo'] else 'causa fisica visible'
        tabla[key][f['real']] += 1
    print(f"  {'evidencia':<28} {'APROBADO':>10} {'RECHAZADO':>10}   {'% rechazo':>10}")
    for k, v in sorted(tabla.items()):
        tot = v['APROBADO'] + v['RECHAZADO']
        print(f"  {k:<28} {v['APROBADO']:>10} {v['RECHAZADO']:>10}   "
              f"{v['RECHAZADO']/max(tot,1)*100:>9.0f}%")

    # ── H2 sobre casos con vegetacion ────────────────────────────────
    veg = [f for f in filas if f['h1_causa_fisica'] and f['punto'] in ('1', '3', '9')]
    print("\n" + "=" * 78)
    print(f"H2 — PRUEBA DE NO-PREVENIBILIDAD  (puntos 1/3/9 con causa fisica, n={len(veg)})")
    print("=" * 78)
    tabla2 = defaultdict(lambda: {'APROBADO': 0, 'RECHAZADO': 0})
    for f in veg:
        if f['h2_medicion'] and f['h2_tercero']:
            key = 'medicion + tercero'
        elif f['h2_medicion']:
            key = 'solo medicion'
        elif f['h2_tercero']:
            key = 'solo tercero documentado'
        else:
            key = 'sin prueba verificable'
        tabla2[key][f['real']] += 1
    print(f"  {'prueba':<28} {'APROBADO':>10} {'RECHAZADO':>10}   {'% aprob':>10}")
    for k, v in sorted(tabla2.items()):
        tot = v['APROBADO'] + v['RECHAZADO']
        print(f"  {k:<28} {v['APROBADO']:>10} {v['RECHAZADO']:>10}   "
              f"{v['APROBADO']/max(tot,1)*100:>9.0f}%")

    # ── regla combinada propuesta ────────────────────────────────────
    print("\n" + "=" * 78)
    print("REGLA COMBINADA PROPUESTA (solo diagnostico, no altera decisiones)")
    print("  RECHAZAR si punto==1 y (solo dispositivo  o  sin prueba verificable)")
    print("=" * 78)
    tp = tn = fp = fn = 0
    for f in clima:
        regla = 'RECHAZADO' if (f['h1_solo_dispositivo'] or
                                not (f['h2_medicion'] or f['h2_tercero'])) else 'APROBADO'
        if f['real'] == 'APROBADO' and regla == 'APROBADO':
            tp += 1
        elif f['real'] == 'RECHAZADO' and regla == 'RECHAZADO':
            tn += 1
        elif f['real'] == 'RECHAZADO' and regla == 'APROBADO':
            fp += 1
        else:
            fn += 1
    n = max(len(clima), 1)
    print(f"  aciertos: {tp+tn}/{n} ({(tp+tn)/n*100:.0f}%)   "
          f"TP={tp} TN={tn} FP={fp} FN={fn}")
    print("\n  Casos donde la regla falla (revisar a mano):")
    for f in clima:
        regla = 'RECHAZADO' if (f['h1_solo_dispositivo'] or
                                not (f['h2_medicion'] or f['h2_tercero'])) else 'APROBADO'
        if regla != f['real']:
            print(f"    {f['caso']:<28} real={f['real']:<10} regla={regla:<10} "
                  f"disp={f['h1_solo_dispositivo']} med={f['h2_medicion']} terc={f['h2_tercero']}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'resultados_inferencia.json')