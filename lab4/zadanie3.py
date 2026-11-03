import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sensor import sensor_init, read_sensor

WINDOW = 5      # tyle pomiarow wchodzi do modelu
MAX_POINTS = 100  # tyle punktow widac na wykresie

# Wczytanie wytrenowanych modeli (bez ponownego trenowania)
linear_model = joblib.load("linear_model.pkl")
tree_model = joblib.load("tree_model.pkl")

vl, dev = sensor_init()

# Listy na dane do wykresu
real_x = []
real_y = []
pred_x = []
linear_y = []
tree_y = []

plt.ion()  # tryb interaktywny - wykres odswieza sie w petli
fig = plt.figure()

step = 0
try:
    while plt.fignum_exists(fig.number):
        # Odczyt rzeczywistej wartosci
        distance, _ = read_sensor(vl, dev)
        real_x.append(step)
        real_y.append(distance)

        # Predykcja kolejnej wartosci, gdy mamy juz 5 pomiarow
        if len(real_y) >= WINDOW:
            last_values = np.array([real_y[-WINDOW:]])
            linear_pred = linear_model.predict(last_values)[0]
            tree_pred = tree_model.predict(last_values)[0]

            # Prognoza dotyczy NASTEPNEGO pomiaru, wiec przesuwamy ja o 1
            pred_x.append(step + 1)
            linear_y.append(linear_pred)
            tree_y.append(tree_pred)

            print(f"Pomiar: {distance} mm | LR nastepny: {linear_pred:.1f} mm | Drzewo nastepny: {tree_pred:.1f} mm")

        # Rysowanie ostatnich MAX_POINTS punktow
        plt.cla()
        plt.plot(real_x[-MAX_POINTS:], real_y[-MAX_POINTS:], label="Rzeczywiste")
        plt.plot(pred_x[-MAX_POINTS:], linear_y[-MAX_POINTS:], label="Regresja liniowa")
        plt.plot(pred_x[-MAX_POINTS:], tree_y[-MAX_POINTS:], label="Drzewo decyzyjne")
        plt.xlabel("Numer pomiaru")
        plt.ylabel("Odleglosc [mm]")
        plt.title("Predykcja odleglosci w czasie rzeczywistym")
        plt.legend(loc="upper left")
        plt.pause(0.2)  # odswiezenie wykresu i opoznienie miedzy pomiarami

        step = step + 1
except KeyboardInterrupt:
    print("Koniec")

plt.ioff()
plt.close()
