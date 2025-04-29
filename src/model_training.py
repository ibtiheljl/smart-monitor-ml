import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

#charger les données
df = pd.read_csv("../data/simulated_data.csv")
X = df[["temp", "humidity"]] #les caracteristiques (inputs)
y = df["label"] #les etiquettes(outputs)

# 2. Séparer en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Créer le modèle et l'entraîner
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Évaluer le modèle
accuracy = model.score(X_test, y_test)
print(f"🎯 Précision du modèle : {accuracy:.2%}")

# 5. Sauvegarder le modèle entraîné
joblib.dump(model, "../models/anomaly_detector.pkl")
print("✅ Modèle sauvegardé dans models/anomaly_detector.pkl")