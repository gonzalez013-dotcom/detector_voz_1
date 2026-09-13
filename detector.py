"""
detector.py
-----------
Extracción de características (features) a partir del audio ya separado
en canales. Dos familias, tal como sugiere el reto:

  1) Acústicas: analizan solo la voz del cliente (canal 0) buscando
     artefactos típicos de síntesis de voz (TTS / voice cloning):
     pitch demasiado estable, ausencia de "ruido" natural de
     respiración, espectro plano, jitter/shimmer anómalos, etc.

  2) Conversacionales: usan AMBOS canales para mirar el ritmo del
     turno de palabra -> cómo reacciona el cliente cuando el agente
     lo interrumpe, se queda en silencio o le habla encima. Un
     humano reacciona rápido pero de forma irregular; un bot/TTS
     reacciona con una latencia y un patrón mucho más consistentes.

No se hace nada de red ni de Flask aquí: esto es análisis puro.
"""

from dataclasses import dataclass, field

import numpy as np
import librosa

FRAME_LENGTH = 400   # 50 ms a 8kHz
HOP_LENGTH = 160      # 20 ms a 8kHz
UMBRAL_VOZ_DB = -35.0  # bajo este nivel (relativo al pico) se considera silencio


# ---------------------------------------------------------------------------
# Utilidades de bajo nivel
# ---------------------------------------------------------------------------

def _rms_db(señal: np.ndarray, sr: int):
    """RMS por frame en dBFS relativo al pico de la señal."""
    rms = librosa.feature.rms(
        y=señal, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    pico = np.max(rms) if rms.size and np.max(rms) > 1e-9 else 1e-9
    return librosa.amplitude_to_db(rms / pico, ref=1.0)


def detectar_actividad(señal: np.ndarray, sr: int) -> np.ndarray:
    """
    VAD (voice activity detection) simple basado en energía.
    Regresa un array booleano por frame: True = hay voz.
    Suficientemente robusto para audio telefónico 8kHz y evita
    dependencias pesadas tipo webrtcvad.
    """
    db = _rms_db(señal, sr)
    activo = db > UMBRAL_VOZ_DB

    # Rellenar huecos muy cortos (<80ms) para no partir una palabra en dos.
    min_hueco_frames = max(1, int(0.08 * sr / HOP_LENGTH))
    activo = _rellenar_huecos_cortos(activo, min_hueco_frames)
    return activo


def _rellenar_huecos_cortos(mask: np.ndarray, min_frames: int) -> np.ndarray:
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            if (j - i) < min_frames and i > 0 and j < n:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def _segmentos(mask: np.ndarray, hop_length: int, sr: int):
    """Convierte una máscara booleana por frame en lista de (inicio_s, fin_s)."""
    segmentos = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            segmentos.append((i * hop_length / sr, j * hop_length / sr))
            i = j
        else:
            i += 1
    return segmentos


# ---------------------------------------------------------------------------
# 1) Características acústicas (solo canal del cliente)
# ---------------------------------------------------------------------------

def features_acusticas(cliente: np.ndarray, sr: int) -> dict:
    if cliente.size < sr * 0.3:
        # Clip demasiado corto para sacar estadísticas confiables.
        return _features_acusticas_vacias()

    f0, voiced_flag, _ = librosa.pyin(
        cliente,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
        frame_length=FRAME_LENGTH * 2,
        hop_length=HOP_LENGTH,
    )
    f0_voiced = f0[voiced_flag] if f0 is not None else np.array([])
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)] if f0_voiced.size else f0_voiced

    if f0_voiced.size > 3:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        # Jitter aproximado: variación relativa ciclo a ciclo del pitch.
        diffs = np.abs(np.diff(f0_voiced))
        jitter = float(np.mean(diffs) / f0_mean) if f0_mean > 0 else 0.0
    else:
        f0_mean, f0_std, jitter = 0.0, 0.0, 0.0

    rms = librosa.feature.rms(
        y=cliente, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    # Shimmer aproximado: variación relativa ciclo a ciclo de la energía.
    shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-9))

    flatness = librosa.feature.spectral_flatness(y=cliente)[0]
    centroid = librosa.feature.spectral_centroid(y=cliente, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(
        cliente, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH
    )[0]
    mfcc = librosa.feature.mfcc(y=cliente, sr=sr, n_mfcc=13)

    # Proxy de "respiración natural": micro-silencios cortos (100-350ms)
    # entre segmentos de voz. El TTS suele o no tener respiraciones o
    # ponerlas de forma demasiado regular/artificial.
    mask_voz = detectar_actividad(cliente, sr)
    huecos = _duraciones_de_huecos(mask_voz, HOP_LENGTH, sr)
    huecos_respiracion = [h for h in huecos if 0.1 <= h <= 0.35]
    ratio_respiracion = len(huecos_respiracion) / max(1, len(huecos))
    variabilidad_pausas = float(np.std(huecos)) if len(huecos) > 2 else 0.0

    return {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "flatness_mean": float(np.mean(flatness)),
        "flatness_std": float(np.std(flatness)),
        "centroid_std": float(np.std(centroid)),
        "zcr_std": float(np.std(zcr)),
        "mfcc_std_mean": float(np.mean(np.std(mfcc, axis=1))),
        "ratio_pausas_respiracion": ratio_respiracion,
        "variabilidad_pausas": variabilidad_pausas,
    }


def _duraciones_de_huecos(mask_voz: np.ndarray, hop_length: int, sr: int):
    silencio = ~mask_voz
    return [fin - ini for ini, fin in _segmentos(silencio, hop_length, sr)]


def _features_acusticas_vacias() -> dict:
    llaves = [
        "f0_mean", "f0_std", "jitter", "shimmer", "flatness_mean",
        "flatness_std", "centroid_std", "zcr_std", "mfcc_std_mean",
        "ratio_pausas_respiracion", "variabilidad_pausas",
    ]
    return {k: 0.0 for k in llaves}


# ---------------------------------------------------------------------------
# 2) Características conversacionales (cliente + agente)
# ---------------------------------------------------------------------------

def features_conversacionales(cliente: np.ndarray, agente: np.ndarray, sr: int) -> dict:
    mask_cliente = detectar_actividad(cliente, sr)
    mask_agente = detectar_actividad(agente, sr)

    n = min(len(mask_cliente), len(mask_agente))
    mask_cliente, mask_agente = mask_cliente[:n], mask_agente[:n]

    seg_agente = _segmentos(mask_agente, HOP_LENGTH, sr)

    # Latencias de respuesta: al terminar cada turno del agente, ¿cuánto
    # tarda el cliente en volver a hablar?
    latencias = []
    for _, fin in seg_agente:
        idx_fin = int(fin * sr / HOP_LENGTH)
        idx_resp = _siguiente_inicio_voz(mask_cliente, idx_fin)
        if idx_resp is not None:
            latencias.append((idx_resp - idx_fin) * HOP_LENGTH / sr)

    if latencias:
        lat_mean = float(np.mean(latencias))
        lat_std = float(np.std(latencias))
    else:
        lat_mean, lat_std = 0.0, 0.0

    # Solapamientos: momentos en que agente y cliente hablan a la vez,
    # es decir, el agente "interrumpe" o habla encima del cliente.
    solapado = mask_cliente & mask_agente
    frames_habla_cliente = max(1, int(np.sum(mask_cliente)))
    ratio_solape = float(np.sum(solapado) / frames_habla_cliente)

    # Recuperación tras interrupción: cuando el agente entra mientras el
    # cliente hablaba, ¿el cliente calla y luego retoma, o sigue igual?
    recuperaciones = _duraciones_recuperacion_tras_interrupcion(
        mask_cliente, mask_agente, HOP_LENGTH, sr
    )
    if recuperaciones:
        rec_mean = float(np.mean(recuperaciones))
        rec_std = float(np.std(recuperaciones))
    else:
        rec_mean, rec_std = 0.0, 0.0

    return {
        "latencia_respuesta_media": lat_mean,
        "latencia_respuesta_std": lat_std,
        "ratio_solape_habla": ratio_solape,
        "recuperacion_media": rec_mean,
        "recuperacion_std": rec_std,
        "num_turnos_agente": float(len(seg_agente)),
    }


def _siguiente_inicio_voz(mask: np.ndarray, desde_idx: int):
    n = len(mask)
    i = max(0, desde_idx)
    while i < n and not mask[i]:
        i += 1
    return i if i < n else None


def _duraciones_recuperacion_tras_interrupcion(mask_cliente, mask_agente, hop_length, sr):
    """
    Busca instantes donde el agente empieza a hablar mientras el cliente
    ya estaba hablando (interrupción) y mide cuánto tarda el cliente en
    volver a tener voz activa de forma sostenida después de eso.
    """
    n = min(len(mask_cliente), len(mask_agente))
    duraciones = []
    i = 1
    while i < n:
        interrumpe = mask_agente[i] and not mask_agente[i - 1] and mask_cliente[i]
        if interrumpe:
            j = _siguiente_inicio_voz(mask_cliente, i)
            # Buscamos el próximo tramo de voz del cliente que dure
            # sostenido, saltando micro-fragmentos residuales.
            k = j
            while k is not None and k < n and mask_cliente[k]:
                k += 1
            if j is not None:
                duraciones.append((j - i) * hop_length / sr)
            i = k if k else i + 1
        else:
            i += 1
    return duraciones


# ---------------------------------------------------------------------------
# API pública: combina ambas familias en un solo vector de features
# ---------------------------------------------------------------------------

ORDEN_FEATURES = [
    # acústicas
    "f0_mean", "f0_std", "jitter", "shimmer", "flatness_mean",
    "flatness_std", "centroid_std", "zcr_std", "mfcc_std_mean",
    "ratio_pausas_respiracion", "variabilidad_pausas",
    # conversacionales
    "latencia_respuesta_media", "latencia_respuesta_std",
    "ratio_solape_habla", "recuperacion_media", "recuperacion_std",
    "num_turnos_agente",
]


def extraer_features(cliente: np.ndarray, agente: np.ndarray, sr: int) -> dict:
    acusticas = features_acusticas(cliente, sr)
    conversacionales = features_conversacionales(cliente, agente, sr)
    return {**acusticas, **conversacionales}


def vectorizar(features: dict) -> np.ndarray:
    """Convierte el dict de features a un vector numpy en orden fijo."""
    return np.array([features.get(k, 0.0) for k in ORDEN_FEATURES], dtype=np.float32)
