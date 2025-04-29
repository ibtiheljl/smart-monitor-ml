import numpy as np
import pandas as pd

n = 500
temp = np.random.normal(25, 3, n) # generer des valeurs autour de 25° avec une variation (ecart-type)+-3
humidity = np.random.normal(50, 10, n)# generer des valeurs autour de 50 avec une variation (ecart-type)+-10
labels = ["normal" if t < 30 else "anomalie" for t in temp] #les etiquettes,Ce sont ces "labels" qui vont servir à entraîner le modèle de Machine Learning.

df = pd.DataFrame({
    "temp": temp,
    "humidity": humidity,
    "label": labels
})

df.to_csv("../data/simulated_data.csv", index=False)
print("✅ Données simulées sauvegardées dans data/simulated_data.csv")
