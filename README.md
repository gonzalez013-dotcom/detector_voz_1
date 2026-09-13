# detector_voz — HackMTY26 / Altur

Prototipo que decide si el llamante de una conversación telefónica
(banco + agente de IA) es un humano real o una voz sintética.

## Enfoque

Combina dos de las tres señales que sugiere el reto:

1. **Acústica** (`detector.py::features_acusticas`), solo sobre el canal
   0 (cliente): estabilidad del pitch (F0), jitter y shimmer
   aproximados, planitud espectral, dispersión de MFCC, y un proxy de
   "respiración natural" a partir de micro-silencios de 100–350 ms
   entre segmentos de voz.

2. **Comportamiento conversacional** (`detector.py::features_conversacionales`),
   usando los dos canales: latencia de respuesta después de cada turno
   del agente, cuánto se solapan cliente y agente, y cómo se recupera
   el cliente cuando el agente lo interrumpe. La idea del reto es
   exactamente esta: un humano recupera el turno de forma rápida pero
   *irregular*; un bot lo hace de forma muy *consistente*, y esa
   consistencia (poca varianza) es la señal.

No se implementó la señal semántica (lo que dice el llamante) en esta
primera versión — queda como extensión natural si da tiempo, usando
las transcripciones o un ASR sobre el canal del cliente.

## Cómo decide (modelo.py)

- **Sin dataset de entrenamiento:** usa una heurística explicable, con
  pesos por feature basados en el conocimiento del dominio (ver
  `PESOS_HEURISTICA`). Sirve de red de seguridad y es fácil de explicar
  a los jueces.
- **Con dataset de entrenamiento:** corre `train.py` sobre el dataset
  de Altur (carpetas `humano/` y `sintetico/` con .wav) para entrenar
  un `RandomForestClassifier` y guardarlo en `modelo.pkl`. Si ese
  archivo existe, `app.py` lo usa automáticamente en vez de la
  heurística — no hay que cambiar nada más.

## Estructura

```
detector_voz/
├── app.py               # servidor Flask, expone POST /detect
├── audio.py              # decodificar base64 / leer WAV / separar canales
├── detector.py            # extracción de features acústicas y conversacionales
├── modelo.py              # heurística + wrapper del modelo entrenado (opcional)
├── train.py               # entrena modelo.pkl si hay dataset etiquetado
├── probar_endpoint.py     # cliente de prueba para mandar un .wav a /detect
├── requirements.txt
└── README.md
```

## Cómo correrlo

```bash
pip install -r requirements.txt
python app.py
# servidor en http://0.0.0.0:5000
```

Probar con un archivo local:

```bash
python probar_endpoint.py ruta/a/llamada.wav
```

Formato de request/respuesta (igual al del reto):

```json
// POST /detect
{ "audio": "<wav estéreo 8kHz en base64>" }

// respuesta
{ "is_synthetic": true, "confidence": 0.87 }
```

## Entrenar el modelo (opcional, recomendado en cuanto haya dataset)

```bash
python train.py --dataset ruta/al/dataset
# genera modelo.pkl junto a app.py
```

## Por qué debería funcionar en producción (feasibility)

- El VAD y las features son puramente numéricas (numpy/scipy/librosa),
  sin llamadas a APIs externas ni modelos pesados de deep learning:
  latencia baja y sin dependencia de conectividad.
- Todo el pipeline opera sobre el WAV completo en una sola pasada; para
  streaming en tiempo real se podría correr sobre ventanas deslizantes
  reutilizando las mismas funciones de `detector.py`.
- La heurística es interpretable feature por feature, lo cual importa
  para un banco: se puede auditar por qué el sistema marcó una llamada
  como sospechosa.

## Limitaciones conocidas / próximos pasos

- El VAD por energía es simple; con ruido de línea fuerte puede fallar.
  Si el dataset trae condiciones muy ruidosas, conviene afinar
  `UMBRAL_VOZ_DB` en `detector.py` o migrar a un VAD más robusto.
- Los rangos de normalización en `modelo.py::RANGOS` son estimaciones
  iniciales; deben calibrarse mirando distribuciones reales del
  dataset de Altur.
- Falta la señal semántica (pedir que el cliente repita datos o
  responda sobre algo inexistente); es la extensión más obvia para
  sumar puntos de originalidad si el tiempo alcanza.
