import time
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sensor import sensor_init, read_sensor

WINDOW = 5      # pomiary wchodzace
MAX_POINTS = 100  # pukty na wykresie

linear_model = joblib.load("linear_model.pkl")
tree_model = joblib.load("tree_model.pkl")

vl, dev = sensor_init()

real_x = []
real_y = []
pred_x = []
linear_y = []
tree_y = []

plt.ion() 
fig = plt.figure()

step = 0
try:
    while plt.fignum_exists(fig.number):
        distance, _ = read_sensor(vl, dev)
        real_x.append(step)
        real_y.append(distance)

        if len(real_y) >= WINDOW:
            last_values = np.array([real_y[-WINDOW:]])
            linear_pred = linear_model.predict(last_values)[0]
            tree_pred = tree_model.predict(last_values)[0]

            pred_x.append(step + 1)
            linear_y.append(linear_pred)
            tree_y.append(tree_pred)

            print(f"Pomiar: {distance} mm | LR nastepny: {linear_pred:.1f} mm | Drzewo nastepny: {tree_pred:.1f} mm")

        plt.cla()
        plt.plot(real_x[-MAX_POINTS:], real_y[-MAX_POINTS:], label="Rzeczywiste")
        plt.plot(pred_x[-MAX_POINTS:], linear_y[-MAX_POINTS:], label="Regresja liniowa")
        plt.plot(pred_x[-MAX_POINTS:], tree_y[-MAX_POINTS:], label="Drzewo decyzyjne")
        plt.xlabel("Numer pomiaru")
        plt.ylabel("Odleglosc [mm]")
        plt.title("Predykcja odleglosci w czasie rzeczywistym")
        plt.legend(loc="upper left")
        plt.pause(0.2)

        step = step + 1
except KeyboardInterrupt:
    print("Koniec")

plt.ioff()
plt.close()
