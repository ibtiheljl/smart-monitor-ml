import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv("../data/simulated_data.csv")
X = df[["temp", "humidity"]]
y = df["label"]

# Affichage des données brutes
plt.figure(figsize=(8, 6))
colors = {"normal": "green", "anomalie": "red"}
plt.scatter(df["temp"], df["humidity"], c=df["label"].map(colors), alpha=0.6)
plt.xlabel("Température (°C)")
plt.ylabel("Humidité (%)")
plt.title("Température vs Humidité (coloré par label)")
plt.grid(True)
plt.show()

# Modèle + matrice de confusion
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=["normal", "anomalie"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["normal", "anomalie"])
disp.plot(cmap="Blues")
plt.title("Matrice de confusion du modèle")
plt.show()
