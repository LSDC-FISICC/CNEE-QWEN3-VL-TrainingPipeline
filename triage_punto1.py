"""
Triage de tres vias para los casos de punto 1, y diagnostico de "cajon de sastre".

RESULTADO DE LA VALIDACION ANTERIOR (climate_validation.py, n=63 punto 1):
  - "solo dispositivo/maniobra"        -> 13/15 RECHAZADO (87%)  = brazo de RECHAZO
  - "medicion o tercero documentado"   -> 12/13 APROBADO  (92%)  = brazo de APROBACION
  - resto (causa fisica sin prueba)    -> 26/17            (60%) = moneda al aire
  La regla binaria fallo porque uso la AUSENCIA de prueba como rechazo; ese bucket
  esta 60% aprobado. Lo correcto es un triage de 3 vias, no una regla binaria.

Este script:
  (1) mide precision y cobertura de cada brazo del triage por separado;
  (2) clasifica los punto-1 por FAMILIA DE CAUSA real (animal, descarga atmosferica,
      tercero, vegetacion, objeto, sin causa) para contrastar la hipotesis de que el
      modelo usa el punto 1 como cajon de sastre;
  (3) amplia los patrones de Canal A que el clasificador anterior no capturaba
      (cepellon, arrancado de raiz, deslizamiento) y el orden de palabras de
      "terceras personas realizaron los cortes".

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
    r'terceras personas realiza\w*\s+(los\s+)?cortes?|terceras personas .{0,30}(cort|tala|poda)|'
    r'cepell[oó]n|arrancad[oa] de ra[ií]z|deslizamiento de terreno|'
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


# ORDER MATTERS: a specific foreign object wins over the generic "aislador"
# that also appears in lightning descriptions.
FAMILIAS = [
    ("animal",            re.compile(r'\b(ave|aves|gato|perro|ardilla|nido|animal)\b', re.I)),
    ("tercero_objeto",    re.compile(r'(fibra óptica|fibra optica|nylon|rótulo|rotulo|lámina|lamina|'
                                     r'objeto laminado|retenida|fusible directo|incendio|cohetillo|'
                                     r'quema de (caña|cana)|vehículo|vehiculo|camión|camion)', re.I)),
    ("descarga_atmosf",   re.compile(r'(descarga\s+(electro)?atmosf|electroatmosf|pararrayo)', re.I)),
    ("vegetacion",        re.compile(r'(árbol|arbol|rama|palma|bambú|bambu|banano|musgo|vegetaci)', re.I)),
]


def familia(parsed):
    """Best-effort cause family from the model's own description."""
    txt = texto_evidencia(parsed) + ' ' + str(parsed.get('causa_identificada', ''))
    for nombre, rx in FAMILIAS:
        if rx.search(txt):
            return nombre
    return "sin_causa_identificada"


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

    # ── TRIAGE DE 3 VIAS ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("TRIAGE DE 3 VIAS SOBRE PUNTO 1 (diagnostico)")
    print("  brazo RECHAZO   : solo dispositivo/maniobra")
    print("  brazo APROBACION: medicion o tercero documentado")
    print("  resto           : escalar a revision manual")
    print("=" * 78)
    rech = [f for f in clima if f['h1_solo_dispositivo']]
    apro = [f for f in clima if not f['h1_solo_dispositivo']
            and (f['h2_medicion'] or f['h2_tercero'])]
    esc = [f for f in clima if f not in rech and f not in apro]
    for nombre, grupo, esperado in (("RECHAZO", rech, 'RECHAZADO'),
                                    ("APROBACION", apro, 'APROBADO'),
                                    ("ESCALAR", esc, None)):
        n = len(grupo)
        if n == 0:
            print(f"  {nombre:<12} vacio")
            continue
        if esperado:
            ok = sum(1 for f in grupo if f['real'] == esperado)
            print(f"  {nombre:<12} n={n:>3} ({n/len(clima)*100:4.0f}% de punto 1) | "
                  f"precision {ok}/{n} = {ok/n*100:.0f}%")
        else:
            a = sum(1 for f in grupo if f['real'] == 'APROBADO')
            print(f"  {nombre:<12} n={n:>3} ({n/len(clima)*100:4.0f}% de punto 1) | "
                  f"reparto {a}A/{n-a}R  (sin señal -> revision manual)")

    # ── familias de causa dentro de punto 1 ──────────────────────────
    print("\n" + "=" * 78)
    print("FAMILIA DE CAUSA REAL DENTRO DE LOS CASOS PUNTO 1")
    print("  (contrasta la hipotesis de 'cajon de sastre')")
    print("=" * 78)
    tabla3 = defaultdict(lambda: {'APROBADO': 0, 'RECHAZADO': 0})
    for r in resultados:
        parsed = reparar_json(r.get('prediccion', ''))
        if parsed is None or str(parsed.get('punto_arbol_aplicado', '')) != '1':
            continue
        tabla3[familia(parsed)][r['label_real']] += 1
    print(f"  {'familia':<26} {'APROBADO':>10} {'RECHAZADO':>10}   {'% aprob':>10}")
    for k, v in sorted(tabla3.items(), key=lambda x: -(x[1]['APROBADO'] + x[1]['RECHAZADO'])):
        tot = v['APROBADO'] + v['RECHAZADO']
        print(f"  {k:<26} {v['APROBADO']:>10} {v['RECHAZADO']:>10}   "
              f"{v['APROBADO']/max(tot,1)*100:>9.0f}%")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'resultados_inferencia.json')