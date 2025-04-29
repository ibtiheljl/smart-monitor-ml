import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import csv
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

style = ttk.Style()
style.theme_use("clam")

# Charger le modèle
model = joblib.load("../models/anomaly_detector.pkl")

# État global pour boucle auto
running = False
history = []

# Fenêtre principale
window = tk.Tk()
window.configure(bg="#f2f2f2")
window.title("Smart Monitor ML")
window.geometry("700x500")
window.resizable(False, False)

# ---------- FRAME GRAPHE ----------
fig = Figure(figsize=(4, 2), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Température et Humidité")
ax.set_xlabel("Lecture #")
ax.set_ylabel("Valeur")
line_temp, = ax.plot([], [], label="Température (°C)", color="red")
line_hum, = ax.plot([], [], label="Humidité (%)", color="blue")
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=False)

temps = []
hums = []

# ---------- FONCTIONS ----------
def export_history():
    if not history:
        messagebox.showinfo("Info", "Aucune donnée à exporter.")
        return

    with open("../data/history.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Température", "Humidité", "État"])
        for entry in history:
            writer.writerow(entry)
    messagebox.showinfo("Succès", "Historique exporté dans data/history.csv")

def simulate():
    temp = round(np.random.normal(25, 3), 2)
    hum = round(np.random.normal(50, 10), 2)
    prediction = model.predict([[temp, hum]])[0]

    temp_label.config(text=f"Température: {temp} °C")
    humidity_label.config(text=f"Humidité: {hum} %")

    if prediction == "normal":
        status_label.config(text="🟢 NORMAL", background="green")
    else:
        status_label.config(text="🔴 ANOMALIE ⚠️", background="red")

    # MAJ historique
    history.append((temp, hum, prediction))
    history_table.insert("", "end", values=(temp, hum, prediction))

    # MAJ graphe
    temps.append(temp)
    hums.append(hum)
    x = list(range(len(temps)))
    line_temp.set_data(x, temps)
    line_hum.set_data(x, hums)
    ax.relim()
    ax.autoscale_view()
    canvas.draw()

def auto_loop():
    if running:
        simulate()
        window.after(2000, auto_loop)

def start_auto():
    global running
    running = True
    auto_loop()

def stop_auto():
    global running
    running = False
    status_label.config(text="⏸️ Arrêté", background="gray")

# ---------- INTERFACE ----------
frame_top = ttk.Frame(window)
frame_top.pack(pady=10)

title = ttk.Label(frame_top, text="Simulation Capteur + IA", font=("Segoe UI", 12, "bold"), foreground="#333")
title.pack()

frame_info = ttk.Frame(window)
frame_info.pack(pady=5)

temp_label = ttk.Label(frame_info, text="Température: -- °C", font=("Arial", 12))
temp_label.grid(row=0, column=0, padx=10)

humidity_label = ttk.Label(frame_info, text="Humidité: -- %", font=("Arial", 12))
humidity_label.grid(row=0, column=1, padx=10)

status_label = tk.Label(frame_info, text="État: --", font=("Arial", 12), width=20, background="gray", fg="white")
status_label.grid(row=0, column=2, padx=10)

frame_buttons = ttk.Frame(window)
frame_buttons.pack(pady=5)

ttk.Button(frame_buttons, text="Lecture manuelle", command=simulate).grid(row=0, column=0, padx=5)
ttk.Button(frame_buttons, text="▶️ Auto", command=start_auto).grid(row=0, column=1, padx=5)
ttk.Button(frame_buttons, text="⏹️ Stop", command=stop_auto).grid(row=0, column=2, padx=5)
ttk.Button(frame_buttons, text="💾 Exporter CSV", command=export_history).grid(row=0, column=3, padx=5)

# ---------- HISTORIQUE ----------
frame_history = ttk.Frame(window)
frame_history.pack(pady=10)

ttk.Label(frame_history, text="Historique des lectures").pack()

history_table = ttk.Treeview(frame_history, columns=("temp", "hum", "status"), show="headings", height=6)
history_table.heading("temp", text="Température")
history_table.heading("hum", text="Humidité")
history_table.heading("status", text="État")

history_table.column("temp", width=100)
history_table.column("hum", width=100)
history_table.column("status", width=120)

history_table.pack()

window.mainloop()
