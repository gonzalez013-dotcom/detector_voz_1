"""
audio.py
--------
Carga de audio y separación de canales.

El reto especifica: WAV estéreo, 8 kHz, telefónico.
  Canal 0 = llamante (cliente)  -> el que hay que clasificar
  Canal 1 = agente (bot)        -> contexto conversacional

Este módulo solo se encarga de E/S de audio: decodificar base64,
leer el WAV, separar canales y normalizar. No hace análisis.
"""

import base64
import io

import numpy as np
import soundfile as sf

CANAL_CLIENTE = 0
CANAL_AGENTE = 1

SR_ESPERADO = 8000  # Hz, telefonía


class AudioInvalido(ValueError):
    """El WAV recibido no cumple el formato esperado (estéreo)."""


def decodificar_base64_wav(b64_str: str) -> bytes:
    """Convierte el string base64 recibido en /detect a bytes crudos de WAV."""
    # Algunos clientes envían el prefijo data URI, lo quitamos si aparece.
    if "," in b64_str[:60] and b64_str.strip().startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)


def cargar_wav_estereo(datos: bytes):
    """
    Lee bytes de un WAV estéreo y regresa (cliente, agente, sr) como
    arrays float32 en rango [-1, 1].
    """
    with io.BytesIO(datos) as buf:
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)

    if audio.shape[1] < 2:
        raise AudioInvalido(
            f"Se esperaba audio estéreo (2 canales), se recibió {audio.shape[1]} canal(es)."
        )

    cliente = audio[:, CANAL_CLIENTE]
    agente = audio[:, CANAL_AGENTE]

    return cliente, agente, sr


def desde_base64(b64_str: str):
    """Atajo: base64 -> (cliente, agente, sr)."""
    datos = decodificar_base64_wav(b64_str)
    return cargar_wav_estereo(datos)


def desde_archivo(ruta: str):
    """Atajo para pruebas locales: lee un .wav del disco."""
    with open(ruta, "rb") as f:
        datos = f.read()
    return cargar_wav_estereo(datos)


def normalizar(señal: np.ndarray) -> np.ndarray:
    """Normaliza amplitud a pico 1.0, evitando dividir entre cero."""
    pico = np.max(np.abs(señal)) if señal.size else 0.0
    if pico < 1e-6:
        return señal
    return señal / pico
