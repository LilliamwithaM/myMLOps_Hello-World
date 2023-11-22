import json
import numpy as np
import requests

# URL del servidor que sirve el modelo
SERVER_URL = 'https://linear-model-service-lilliamwitham.cloud.okteto.net/v1/models/linear-model:predict'

def main():
    print("Fórmula: y = -4x + 1")

    # Datos para predecir
    instances = [[0.0], [1.0], [2.0]]

    # Crear la solicitud de predicción
    predict_request = {
        'instances': instances
    }

    # Enviar la solicitud de predicción al servidor utilizando la opción 'json'
    response = requests.post(SERVER_URL, json=predict_request)

    # Verificar si la solicitud fue exitosa
    response.raise_for_status()

    # Obtener las predicciones del servidor
    predictions = response.json()

    # Imprimir las predicciones
    print('Predictions:', predictions)

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, json=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()

    print('Prediction class: {}, avg latency: {} ms'.format(
        np.argmax(prediction), (total_time * 1000) / num_requests))

if __name__ == '__main__':
    main()
