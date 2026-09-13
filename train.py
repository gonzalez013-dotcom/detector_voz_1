"""
train.py
--------
Entrena un RandomForest sobre el dataset de Altur y guarda modelo.pkl.
Uso opcional: si no lo corres, app.py sigue funcionando con la
heurística de modelo.py.

Estructura esperada (la que trae el dataset de Altur):

    dataset/
        audio/
            call_0181ce113ebe.wav
            call_018c9d3823ac.wav
            ...
        manifest.csv        <- columnas: anon_id,label,split,duration_s
                                label es "human" o "synthetic"
                                split es "train" o "val"

Uso:
    python train.py --dataset ruta/al/dataset

Por defecto usa la partición train/val que ya viene en el manifest
(en vez de partir los datos al azar), para que la validación sea
comparable a como se organizó el dataset original.

La extracción de features (lo más lento del proceso, sobre todo
librosa.pyin) se cachea en features_cache.csv dentro de la carpeta
del dataset, así que si vuelves a correr train.py (por ejemplo tras
ajustar el modelo) no hay que re-procesar el audio desde cero.
"""

import argparse
import csv
import os
import time

import numpy as np

from audio import desde_archivo
from detector import extraer_features, ORDEN_FEATURES

ETIQUETAS = {"human": 0, "synthetic": 1}


def leer_manifest(ruta_manifest):
    filas = []
    with open(ruta_manifest, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            filas.append(fila)
    return filas


def cargar_cache(ruta_cache):
    cache = {}
    if not os.path.exists(ruta_cache):
        return cache
    with open(ruta_cache, newline="", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            anon_id = fila["anon_id"]
            cache[anon_id] = [float(fila[k]) for k in ORDEN_FEATURES]
    return cache


def guardar_cache(ruta_cache, cache):
    with open(ruta_cache, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["anon_id"] + ORDEN_FEATURES)
        for anon_id, vector in cache.items():
            escritor.writerow([anon_id] + list(vector))


def extraer_o_reusar(anon_id, ruta_wav, cache):
    if anon_id in cache:
        return cache[anon_id]
    cliente, agente, sr = desde_archivo(ruta_wav)
    features = extraer_features(cliente, agente, sr)
    vector = [features.get(k, 0.0) for k in ORDEN_FEATURES]
    cache[anon_id] = vector
    return vector


def construir_dataset(ruta_dataset, ruta_manifest, cache):
    carpeta_audio = os.path.join(ruta_dataset, "audio")
    filas = leer_manifest(ruta_manifest)

    X_train, y_train, X_val, y_val = [], [], [], []
    faltantes, procesados = 0, 0
    inicio = time.time()

    for i, fila in enumerate(filas, start=1):
        anon_id = fila["anon_id"]
        label = fila["label"].strip().lower()
        split = fila["split"].strip().lower()

        if label not in ETIQUETAS:
            print(f"  aviso: etiqueta desconocida '{label}' en {anon_id}, se omite.")
            continue

        ruta_wav = os.path.join(carpeta_audio, f"{anon_id}.wav")
        if not os.path.exists(ruta_wav):
            faltantes += 1
            continue

        try:
            vector = extraer_o_reusar(anon_id, ruta_wav, cache)
        except Exception as e:
            print(f"  aviso: no se pudo procesar {anon_id}: {e}")
            continue

        destino_X, destino_y = (X_train, y_train) if split == "train" else (X_val, y_val)
        destino_X.append(vector)
        destino_y.append(ETIQUETAS[label])
        procesados += 1

        if i % 25 == 0 or i == len(filas):
            transcurrido = time.time() - inicio
            print(f"  procesadas {i}/{len(filas)} llamadas ({transcurrido:.0f}s)...")

    if faltantes:
        print(f"  aviso: {faltantes} llamadas del manifest no se encontraron en {carpeta_audio}")

    print(f"Total usable: {procesados} llamadas ({len(X_train)} train, {len(X_val)} val)")

    return (
        np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.int32),
        np.array(X_val, dtype=np.float32), np.array(y_val, dtype=np.int32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Carpeta con audio/ y manifest.csv")
    parser.add_argument("--manifest", default=None, help="Ruta al manifest.csv (por defecto: <dataset>/manifest.csv)")
    parser.add_argument("--salida", default="modelo.pkl")
    args = parser.parse_args()

    ruta_manifest = args.manifest or os.path.join(args.dataset, "manifest.csv")
    ruta_cache = os.path.join(args.dataset, "features_cache.csv")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import joblib

    cache = cargar_cache(ruta_cache)
    print(f"Features en cache: {len(cache)}")

    X_train, y_train, X_val, y_val = construir_dataset(args.dataset, ruta_manifest, cache)

    guardar_cache(ruta_cache, cache)
    print(f"Cache actualizado en: {ruta_cache}")

    if len(X_train) < 10:
        print("Muy pocos ejemplos de entrenamiento; revisa el dataset y el manifest.")
        return

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    if len(X_val) > 0:
        print("\nReporte en el split 'val' del manifest:")
        print(classification_report(y_val, clf.predict(X_val), target_names=["human", "synthetic"]))
    else:
        print("\n(no había filas 'val' en el manifest, no se evaluó por separado)")

    joblib.dump(clf, args.salida)
    print(f"\nModelo guardado en: {args.salida}")


if __name__ == "__main__":
    main()
