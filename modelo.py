"""
modelo.py
---------
Clasificador final: decide is_synthetic + confidence a partir del
vector de features de detector.py.

Dos modos, para que el prototipo funcione DESDE EL PRIMER MOMENTO
(sin dataset de entrenamiento a mano) pero mejore en cuanto haya
datos etiquetados:

  1) Heurística (fallback, siempre disponible): puntuación ponderada
     basada en el conocimiento del dominio descrito en el reto
     (pitch demasiado estable, poca respiración, pausas muy
     regulares, recuperación tras interrupción demasiado uniforme).

  2) Modelo entrenado (modelo.pkl, opcional): un RandomForest de
     scikit-learn entrenado con train.py sobre el dataset de Altur.
     Si el archivo existe, se usa en vez de la heurística.

Esto es intencional: la heurística es explicable ante los jueces
("por qué funciona") y sirve de red de seguridad si el entrenamiento
no da tiempo; el modelo entrenado sube el desempeño cuando sí hay
dataset.
"""

import os

import numpy as np

from detector import ORDEN_FEATURES

RUTA_MODELO = os.path.join(os.path.dirname(__file__), "modelo.pkl")

# Pesos de la heurística: signo indica si un valor ALTO empuja hacia
# "sintético" (+) o hacia "humano" (-). Elegidos a partir de la
# intuición del reto, no de un fit estadístico; se pueden ajustar
# mirando el dataset real.
PESOS_HEURISTICA = {
    "f0_std": -1.2,                     # pitch muy estable -> sintético
    "jitter": -1.0,                     # jitter natural bajo -> sintético
    "shimmer": -0.6,
    "flatness_mean": 1.0,               # espectro plano -> sintético
    "variabilidad_pausas": -1.0,        # pausas muy regulares -> sintético
    "ratio_pausas_respiracion": -0.8,   # sin respiraciones naturales -> sintético
    "latencia_respuesta_std": -1.3,     # latencia de respuesta muy uniforme -> sintético
    "recuperacion_std": -1.0,           # recuperación tras interrupción uniforme -> sintético
}

# Rango esperable aproximado de cada feature (para normalizar a [0,1]
# antes de aplicar el peso). Ajustables tras ver el dataset real.
RANGOS = {
    "f0_std": (0, 40),
    "jitter": (0, 0.05),
    "shimmer": (0, 0.6),
    "flatness_mean": (0, 0.3),
    "variabilidad_pausas": (0, 0.4),
    "ratio_pausas_respiracion": (0, 1),
    "latencia_respuesta_std": (0, 1.2),
    "recuperacion_std": (0, 1.0),
}


def _normalizar(valor, lo, hi):
    if hi <= lo:
        return 0.0
    return float(np.clip((valor - lo) / (hi - lo), 0.0, 1.0))


class Clasificador:
    def __init__(self):
        self.modelo_ml = self._cargar_modelo_ml()

    def _cargar_modelo_ml(self):
        if os.path.exists(RUTA_MODELO):
            try:
                import joblib
                return joblib.load(RUTA_MODELO)
            except Exception:
                # Si algo falla al cargar, seguimos con la heurística
                # en vez de tumbar el endpoint.
                return None
        return None

    def predecir(self, features: dict):
        if self.modelo_ml is not None:
            return self._predecir_ml(features)
        return self._predecir_heuristica(features)

    def _predecir_ml(self, features: dict):
        x = np.array([[features.get(k, 0.0) for k in ORDEN_FEATURES]], dtype=np.float32)
        proba = self.modelo_ml.predict_proba(x)[0]
        # Se asume que la clase 1 == "sintético" (ver train.py)
        confianza_sintetico = float(proba[1])
        return confianza_sintetico >= 0.5, confianza_sintetico

    def _predecir_heuristica(self, features: dict):
        puntaje = 0.0
        peso_total = 0.0
        for llave, peso in PESOS_HEURISTICA.items():
            lo, hi = RANGOS[llave]
            valor_norm = _normalizar(features.get(llave, 0.0), lo, hi)
            # Si el peso es negativo, un valor alto de la feature resta
            # puntaje "sintético" (o sea, un valor BAJO lo suma).
            aporte = (1 - valor_norm) if peso < 0 else valor_norm
            puntaje += abs(peso) * aporte
            peso_total += abs(peso)

        confianza_sintetico = puntaje / peso_total if peso_total > 0 else 0.5
        return confianza_sintetico >= 0.5, float(confianza_sintetico)


_clasificador_singleton = None


def obtener_clasificador() -> Clasificador:
    global _clasificador_singleton
    if _clasificador_singleton is None:
        _clasificador_singleton = Clasificador()
    return _clasificador_singleton
