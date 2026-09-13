"""
app.py
------
Expone el único endpoint requerido por el reto:

    POST /detect
    body: {"audio": "<wav estéreo 8kHz en base64>"}
    respuesta: {"is_synthetic": bool, "confidence": float}

Mantenlo corriendo y alcanzable durante la ventana de evaluación.
"""

import time
import traceback

import numpy as np
from flask import Flask, request, jsonify, render_template

from audio import desde_base64, AudioInvalido
from detector import extraer_features
from modelo import obtener_clasificador

app = Flask(__name__)
clasificador = obtener_clasificador()


def calentar_pipeline():
    """
    Fuerza la compilación JIT de numba/librosa ANTES de recibir tráfico real.

    La primera vez que se llama a librosa.pyin (u otras funciones que usan
    numba) dentro de un proceso, numba compila código internamente y eso
    puede tardar 20-30+ segundos. El reto da solo 30s por llamada, así que
    si esa primera compilación ocurre durante la llamada real del juez,
    perdemos por timeout. Aquí la forzamos con un audio falso, en memoria,
    apenas arranca el servidor -- antes de que el juez pueda tocarnos.
    """
    try:
        inicio = time.time()
        sr = 8000
        t = np.linspace(0, 3, sr * 3, endpoint=False).astype(np.float32)
        cliente = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        agente = (0.2 * np.sin(2 * np.pi * 140 * t)).astype(np.float32)

        features = extraer_features(cliente, agente, sr)
        clasificador.predecir(features)

        print(f"[warmup] listo en {time.time() - inicio:.1f}s — el servidor ya está caliente.")
    except Exception:
        print("[warmup] aviso: falló el precalentamiento, la primera llamada real podría tardar más de lo normal.")
        traceback.print_exc()


@app.route("/", methods=["GET"])
def pagina_principal():
    """Interfaz web para probar el detector subiendo un WAV a mano."""
    return render_template("index.html")


@app.route("/api/salud", methods=["GET"])
def salud():
    return jsonify({"status": "ok", "servicio": "detector_voz"})


@app.route("/detect", methods=["POST"])
def detectar():
    inicio = time.time()
    payload = request.get_json(silent=True) or {}
    call_id = payload.get("call_id", "?")

    audio_b64 = payload.get("audio_base64") or payload.get("audio") or payload.get("wav")
    if not audio_b64:
        return jsonify({
            "error": "Falta el campo 'audio' con el WAV en base64."
        }), 400

    try:
        cliente, agente, sr = desde_base64(audio_b64)
    except AudioInvalido as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "No se pudo decodificar el WAV recibido."}), 400

    try:
        features = extraer_features(cliente, agente, sr)
        es_sintetico, confianza = clasificador.predecir(features)
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Fallo al analizar el audio."}), 500

    latencia_ms = int((time.time() - inicio) * 1000)

    # OJO: el contrato del juez espera "confidence" = qué tan seguro
    # está el sistema de SU PROPIO veredicto (confidence_endpoint.py
    # hace: confidence si is_synthetic, si no 1-confidence, para
    # reconstruir P(sintético) y calcular AUC/brier). Nuestro
    # clasificador internamente calcula P(sintético) sin importar el
    # veredicto; hay que convertirla aquí antes de responder.
    prob_sintetico = float(confianza)
    confianza_veredicto = prob_sintetico if es_sintetico else (1 - prob_sintetico)

    respuesta = {
        "is_synthetic": bool(es_sintetico),
        "confidence": round(confianza_veredicto, 4),
    }

    # La interfaz web pide detalle=1 para mostrar el desglose de features;
    # el contrato oficial del reto (is_synthetic + confidence) no cambia.
    if request.args.get("detalle") == "1":
        respuesta["features"] = {k: round(float(v), 6) for k, v in features.items()}

    print(f"[/detect] call_id={call_id} is_synthetic={respuesta['is_synthetic']} confidence={respuesta['confidence']} (prob_sintetico={round(prob_sintetico,4)})  ({latencia_ms} ms)")
    return jsonify(respuesta)


if __name__ == "__main__":
    calentar_pipeline()
    # threaded=True: si una llamada tarda, las siguientes no se quedan
    # haciendo fila detrás de ella (el server de desarrollo de Flask
    # por defecto atiende una petición a la vez).
    # host 0.0.0.0 para que sea alcanzable por los jueces en la red del hackathon.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
