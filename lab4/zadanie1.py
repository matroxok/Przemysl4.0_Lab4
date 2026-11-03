import time
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sensor import sensor_init, read_sensor  # Funkcje z poprzedniego laboratorium


# ---------- Czesc 1: zbieranie danych ----------

def collect_data():
    vl, dev = sensor_init()  # Inicjalizacja czujnika
    data = []  # Lista do przechowywania danych

    # Rozpoczecie pomiaru czasu - skala czasowa dla eksperymentu
    start_time = time.time()
    for _ in range(100):
        # Odczyt wartosci odleglosci z czujnika
        distance, _ = read_sensor(vl, dev)
        # Obliczenie aktualnego momentu eksperymentu
        experiment_time = time.time() - start_time
        # Dodanie danych do listy
        data.append([experiment_time, distance])
        # Wyswietlenie danych w konsoli dla monitorowania
        print(f"Time: {experiment_time:.2f}, Distance: {distance} mm")
        # Opoznienie miedzy kolejnymi odczytami
        time.sleep(0.2)

    # Tworzenie DataFrame z danych
    df = pd.DataFrame(data, columns=["time", "distance"])
    # Zapisanie danych do pliku CSV
    df.to_csv("data.csv", index=False)
    return df


# ---------- Czesc 2: trenowanie modelu regresji liniowej ----------

def train_linear():
    df = pd.read_csv("data.csv")
    values = df["distance"].to_numpy()

    WINDOW = 5
    TRAIN_SIZE = 80

    X, y = [], []

    # Przesuwajace sie okno: 5 pomiarow -> 6. pomiar
    for i in range(len(values) - WINDOW):
        X.append(values[i:i + WINDOW])
        y.append(values[i + WINDOW])

    X = np.array(X)
    y = np.array(y)

    # Podzial chronologiczny: 75 okien treningowych,
    # 20 okien testowych.
    split = TRAIN_SIZE - WINDOW

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results = pd.DataFrame({"Rzeczywista wartosc": y_test, "Predykcja": predictions})
    print(results)
    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    joblib.dump(model, "linear_model.pkl")


if __name__ == "__main__":
    collect_data()
    train_linear()
