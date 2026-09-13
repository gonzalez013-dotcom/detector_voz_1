"""
inspeccionar_modelo.py
------------------------
Imprime qué tan importante fue cada feature para el RandomForest
entrenado en modelo.pkl. Sirve como chequeo de sanidad: si UNA sola
feature domina con muchísima diferencia sobre las demás, vale la pena
sospechar que el modelo encontró un atajo (algo incidental del
dataset) en vez de una señal real y generalizable.

Uso:
    python inspeccionar_modelo.py
"""

import joblib
from detector import ORDEN_FEATURES

modelo = joblib.load("modelo.pkl")
importancias = modelo.feature_importances_

pares = sorted(zip(ORDEN_FEATURES, importancias), key=lambda x: x[1], reverse=True)

print("Importancia de cada feature (de mayor a menor):\n")
for nombre, imp in pares:
    barra = "#" * int(imp * 60)
    print(f"{nombre:28s} {imp:.4f}  {barra}")
