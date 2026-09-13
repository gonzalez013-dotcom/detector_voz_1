"""
probar_endpoint.py
-------------------
Utilidad de línea de comandos para probar POST /detect con un .wav local,
sin tener que armar el request a mano.

Uso:
    python probar_endpoint.py ruta/a/llamada.wav
    python probar_endpoint.py ruta/a/llamada.wav --url http://localhost:5000/detect
"""

import argparse
import base64
import json

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav", help="Ruta al archivo WAV estéreo a probar")
    parser.add_argument("--url", default="http://localhost:5000/detect")
    args = parser.parse_args()

    with open(args.wav, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")

    resp = requests.post(args.url, json={"audio": b64}, timeout=30)
    print("Status:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
