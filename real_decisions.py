"""
Recomputa la decisión final y el confidence real basándose en las REGLAS
definidas en el prompt, ignorando el confidence alucinado por el modelo.

REGLAS DE DECISIÓN (del prompt original):
- Si causa_fuerza_mayor.cumple = false → RECHAZADO
- Si causa_fuerza_mayor.cumple = true Y criterios_cumplidos >= 5 → APROBADO
- En cualquier otro caso → RECHAZADO

CONFIDENCE BASADO EN REGLAS:
- APROBADO 7/7 criterios → 0.95
- APROBADO 6/7 criterios → 0.82
- APROBADO 5/7 criterios → 0.65 (zona gris)
- RECHAZADO causa_FM=false → 0.90
- RECHAZADO causa_FM=true pero <5 cumple → 0.55 (zona gris)

ESCALACIÓN A REVISIÓN:
- confidence_real < 0.75 → REQUIERE_REVISION_MANUAL
"""

import json
import sys
from pathlib import Path

# ─── Configuración ──────────────────────────────────────────────────────────
INPUT_PATH = "/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_4b_v3/metricas_validacion.json"
OUTPUT_PATH = "/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_4b_v3/metricas_post_reglas.json"

THRESHOLD_REVISION = 0.75    # Por debajo de esto → REQUIERE_REVISION_MANUAL


# ─── Lógica de cálculo ──────────────────────────────────────────────────────
def calcular_decision_y_confidence(prediccion_texto):
    """
    Calcula la decisión final y el confidence real basado en reglas.
    Returns: dict con decision_regla, confidence_real, criterios_cumplidos_real,
             causa_fm_cumple, decision_final (con escalación), motivo
    """
    # Caso 1: JSON inválido o vacío
    try:
        parsed = json.loads(prediccion_texto)
    except (json.JSONDecodeError, TypeError):
        return {
            'decision_regla': 'INDETERMINADO',
            'confidence_real': 0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple': None,
            'decision_final': 'REQUIERE_REVISION_MANUAL',
            'motivo': 'json_invalido_o_truncado',
        }

    # Caso 2: Sin estructura de criterios
    criterios = parsed.get('criterios', {})
    if not criterios:
        return {
            'decision_regla': 'INDETERMINADO',
            'confidence_real': 0.0,
            'criterios_cumplidos_real': 0,
            'causa_fm_cumple': None,
            'decision_final': 'REQUIERE_REVISION_MANUAL',
            'motivo': 'sin_criterios',
        }

    # Re-contar criterios cumplidos (no confiar en el campo del modelo)
    criterios_cumplidos_real = sum(
        1 for c in criterios.values() if c.get('cumple') is True
    )

    # Obtener causa_fuerza_mayor.cumple (criterio crítico)
    causa_fm = criterios.get('causa_fuerza_mayor', {})
    causa_fm_cumple = causa_fm.get('cumple')

    # ── Aplicar reglas de decisión del prompt ──
    if causa_fm_cumple is False:
        decision_regla = 'RECHAZADO'
        confidence_real = 0.90       # Rechazo categórico
        motivo_decision = 'causa_FM_no_cumple'

    elif causa_fm_cumple is True and criterios_cumplidos_real >= 5:
        decision_regla = 'APROBADO'

        # Confidence escalado por criterios cumplidos
        if criterios_cumplidos_real == 7:
            confidence_real = 0.95
            motivo_decision = 'aprobado_7de7'
        elif criterios_cumplidos_real == 6:
            confidence_real = 0.82
            motivo_decision = 'aprobado_6de7'
        else:  # 5
            confidence_real = 0.65
            motivo_decision = 'aprobado_5de7_zona_gris'

    elif causa_fm_cumple is True and criterios_cumplidos_real < 5:
        decision_regla = 'RECHAZADO'
        confidence_real = 0.55        # Rechazo por documentación insuficiente
        motivo_decision = 'rechazado_causa_ok_pero_docs_insuficientes'

    else:
        # causa_fm_cumple es None (no determinable)
        decision_regla = 'INDETERMINADO'
        confidence_real = 0.0
        motivo_decision = 'causa_FM_no_determinable'

    # ── Aplicar escalación a revisión manual ──
    if decision_regla == 'INDETERMINADO':
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final = motivo_decision
    elif confidence_real < THRESHOLD_REVISION:
        decision_final = 'REQUIERE_REVISION_MANUAL'
        motivo_final = f'{motivo_decision}_confidence_bajo'
    else:
        decision_final = decision_regla
        motivo_final = motivo_decision

    return {
        'decision_regla': decision_regla,
        'confidence_real': round(confidence_real, 2),
        'criterios_cumplidos_real': criterios_cumplidos_real,
        'causa_fm_cumple': causa_fm_cumple,
        'decision_final': decision_final,
        'motivo': motivo_final,
    }


# ─── Procesamiento del archivo ──────────────────────────────────────────────
def main():
    print(f"Leyendo: {INPUT_PATH}")
    with open(INPUT_PATH, encoding='utf-8') as f:
        metricas = json.load(f)

    resultados_nuevos = []
    contadores = {
        'APROBADO': 0,
        'RECHAZADO': 0,
        'REQUIERE_REVISION_MANUAL': 0,
    }
    correctos_auto = 0
    incorrectos_auto = 0
    escalados = 0

    # Tabla de salida
    header = f"{'CASO':<28} {'REAL':<11} {'MODELO':<11} {'NUEVA':<25} {'C':<3} {'CONF':<5} {'MOTIVO'}"
    print('\n' + '=' * 130)
    print(header)
    print('=' * 130)

    for r in metricas['resultados']:
        analisis = calcular_decision_y_confidence(r['prediccion'])

        decision_final = analisis['decision_final']
        confidence_real = analisis['confidence_real']
        criterios = analisis['criterios_cumplidos_real']
        label_real = r['label_real']

        contadores[decision_final] = contadores.get(decision_final, 0) + 1

        # Clasificar resultado
        if decision_final == 'REQUIERE_REVISION_MANUAL':
            escalados += 1
            estado_visual = 'ESC'
        elif decision_final == label_real:
            correctos_auto += 1
            estado_visual = 'OK'
        else:
            incorrectos_auto += 1
            estado_visual = 'FAIL'

        print(f"{r['caso']:<28} {label_real:<11} {r['label_pred']:<11} "
              f"{decision_final:<25} {criterios}/7 {confidence_real:<5} {analisis['motivo']}")

        resultados_nuevos.append({
            'caso': r['caso'],
            'label_real': label_real,
            'label_pred_original': r['label_pred'],
            'decision_regla': analisis['decision_regla'],
            'decision_final': decision_final,
            'confidence_real': confidence_real,
            'criterios_cumplidos_real': criterios,
            'causa_fm_cumple': analisis['causa_fm_cumple'],
            'motivo': analisis['motivo'],
            'estado_post_reglas': estado_visual,
        })

    n = len(resultados_nuevos)

    # ── Resumen final ──
    print('\n' + '=' * 60)
    print('RESUMEN POST-REGLAS')
    print('=' * 60)
    print(f"Total casos:                {n}")
    print(f"  → APROBADO automático:    {contadores.get('APROBADO', 0)}")
    print(f"  → RECHAZADO automático:   {contadores.get('RECHAZADO', 0)}")
    print(f"  → REQUIERE_REVISION:      {contadores.get('REQUIERE_REVISION_MANUAL', 0)}")
    print()
    print(f"De los automáticos ({n - escalados}):")
    if (n - escalados) > 0:
        print(f"  Correctos:  {correctos_auto}/{n - escalados} "
              f"({correctos_auto / (n - escalados) * 100:.1f}%)")
        print(f"  Incorrectos: {incorrectos_auto}/{n - escalados} "
              f"({incorrectos_auto / (n - escalados) * 100:.1f}%)")

    print(f"\nMétricas globales:")
    print(f"  Tasa automatización segura: {correctos_auto / n * 100:.1f}%")
    print(f"  Tasa falsos auto:           {incorrectos_auto / n * 100:.1f}%")
    print(f"  Tasa escalación:            {escalados / n * 100:.1f}%")

    # Comparación con resultado original
    correctos_originales = sum(
        1 for r in metricas['resultados'] if r['label_pred'] == r['label_real']
    )
    print(f"\nComparación con modelo original:")
    print(f"  Original (sin reglas):      {correctos_originales}/{n} "
          f"({correctos_originales / n * 100:.1f}%) accuracy")
    print(f"  Post-reglas:                {correctos_auto}/{n} "
          f"({correctos_auto / n * 100:.1f}%) auto + {escalados} escalados")

    # Guardar resultado completo
    output = {
        'config': {
            'threshold_revision': THRESHOLD_REVISION,
            'reglas_aplicadas': 'prompt_original_CNEE',
        },
        'resumen': {
            'total': n,
            'automatico_aprobado': contadores.get('APROBADO', 0),
            'automatico_rechazado': contadores.get('RECHAZADO', 0),
            'requiere_revision': contadores.get('REQUIERE_REVISION_MANUAL', 0),
            'auto_correctos': correctos_auto,
            'auto_incorrectos': incorrectos_auto,
            'escalados': escalados,
            'tasa_auto_segura': round(correctos_auto / n * 100, 2),
            'tasa_falsos_auto': round(incorrectos_auto / n * 100, 2),
            'tasa_escalacion': round(escalados / n * 100, 2),
        },
        'resultados': resultados_nuevos,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()