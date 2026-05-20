import json
with open("/home/nvidia-ott/lsdc/cnee-native/output/qwen3vl_4b_v3/metricas_dataset_completo.json") as f:
    data = json.load(f)

# Solo errores automaticos
errores = [r for r in data['resultados'] if r['es_correcto_auto'] is False]
print(f"Total errores auto: {len(errores)}")

# Agrupar por causa
from collections import Counter
causas = Counter()
for e in errores:
    # Extraer causa_identificada del JSON predicho
    try:
        p = json.loads(e['prediccion'])
        causa = p.get('causa_identificada', 'SIN_CAUSA')
    except:
        causa = 'JSON_INVALIDO'
    tipo = f"{e['label_real']}->{e['decision_final']}"
    causas[(tipo, causa[:40])] += 1

for (tipo, causa), n in causas.most_common():
    print(f"  {n}x  {tipo:<22} | {causa}")