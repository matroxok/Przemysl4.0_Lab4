import pandas as pd
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

TRAIN_SIZE = 80


def tree_prediction(window, horizon):
    df = pd.read_csv("data.csv")
    values = df["distance"].to_numpy()

    # Przesuwajace sie okno: "window" pomiarow -> kolejny pomiar
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i:i + window])
        y.append(values[i + window])

    X = np.array(X)
    y = np.array(y)

    # Taki sam podzial chronologiczny jak w zadaniu 1
    split = TRAIN_SIZE - window
    X_train = X[:split]
    y_train = y[:split]

    tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    tree.fit(X_train, y_train)

    # Predykcja "horizon" krokow w przod.
    # Kazda przewidziana wartosc jest dopisywana do okna
    # i uzywana do kolejnej prognozy.
    real_values = []
    predicted_values = []

    for i in range(split, len(values) - window - horizon + 1):
        window_values = list(values[i:i + window])

        for step in range(horizon):
            last_values = np.array([window_values[-window:]])
            prediction = tree.predict(last_values)[0]
            window_values.append(prediction)

        # Porownujemy ostatnia przewidziana wartosc z rzeczywista
        predicted_values.append(window_values[-1])
        real_values.append(values[i + window + horizon - 1])

    results = pd.DataFrame({"Rzeczywista wartosc": real_values, "Predykcja": predicted_values})
    print(results)
    mse = mean_squared_error(real_values, predicted_values)

    # Model z oknem 5 zapisujemy do uzycia w zadaniu 3
    if window == 5:
        joblib.dump(tree, "tree_model.pkl")

    return mse


if __name__ == "__main__":
    windows = [5, 10]
    horizons = [1, 3, 5]
    summary = []

    for window in windows:
        for horizon in horizons:
            print(f"\n=== Okno: {window}, horyzont: {horizon} ===")
            mse = tree_prediction(window, horizon)
            print("MSE:", mse)
            summary.append([window, horizon, mse])

    print("\n=== Podsumowanie ===")
    summary_df = pd.DataFrame(summary, columns=["Okno", "Horyzont", "MSE"])
    print(summary_df)
