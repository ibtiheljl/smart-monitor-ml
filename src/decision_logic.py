import joblib
import numpy as np
import time

# Charger le modèle
model = joblib.load("../models/anomaly_detector.pkl")

print("✅ Modèle chargé.")

def simulate_sensor_reading():
    temp = np.random.normal(25, 3)
    humidity = np.random.normal(50, 10)
    return round(temp, 2), round(humidity, 2)

while True:
    temp, humidity = simulate_sensor_reading()
    prediction = model.predict([[temp, humidity]])[0]

    print(f"\n📡 Temp: {temp} °C | Humidité: {humidity} %")
    if prediction == "normal":
        print("🟢 État : NORMAL")
    else:
        print("🔴 État : ANOMALIE ⚠️")

    time.sleep(2)  # attendre 2 secondes avant la prochaine simulation
