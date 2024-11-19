import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(url, usecols=["Passengers"])
data = data.values.astype("float32")


scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)


def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i: i + look_back, 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)


look_back = 12  # Окно (12 месяцев)
train_size = int(len(data_normalized) * 0.8)
train, test = data_normalized[:train_size], data_normalized[train_size:]

X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def build_and_train_model(model_type, neurons, layers, look_back, x_train, y_train, x_test, y_test):
    model = Sequential()
    for i in range(layers):
        if i == layers - 1:  # Последний слой без return_sequences
            if model_type == "RNN":
                model.add(SimpleRNN(neurons, input_shape=(look_back, 1)))
            elif model_type == "LSTM":
                model.add(LSTM(neurons, input_shape=(look_back, 1)))
            elif model_type == "GRU":
                model.add(GRU(neurons, input_shape=(look_back, 1)))
        else:  # Слои с return_sequences=True
            if model_type == "RNN":
                model.add(SimpleRNN(neurons, input_shape=(look_back, 1), return_sequences=True))
            elif model_type == "LSTM":
                model.add(LSTM(neurons, input_shape=(look_back, 1), return_sequences=True))
            elif model_type == "GRU":
                model.add(GRU(neurons, input_shape=(look_back, 1), return_sequences=True))

    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    history = model.fit(
        x_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(x_test, y_test),
        verbose=0
    )
    return model, history.history["val_loss"][-1]


def stepwise_prediction(model, data, look_back):
    predictions = []
    input_seq = data[:look_back]
    for i in range(len(data) - look_back):
        input_seq_reshaped = np.reshape(input_seq, (1, look_back, 1))
        pred = model.predict(input_seq_reshaped, verbose=0)[0, 0]
        predictions.append(pred)
        input_seq = np.append(input_seq[1:], data[look_back + i])
    return np.array(predictions)


configs = [
    {"model_type": "RNN", "neurons": 24, "layers": 1},
    {"model_type": "RNN", "neurons": 32, "layers": 1},
    {"model_type": "LSTM", "neurons": 64, "layers": 1},
    {"model_type": "GRU", "neurons": 64, "layers": 1},
    {"model_type": "LSTM", "neurons": 64, "layers": 2},
    {"model_type": "GRU", "neurons": 64, "layers": 2},
    {"model_type": "LSTM", "neurons": 128, "layers": 2},
    {"model_type": "GRU", "neurons": 128, "layers": 2},
]

# Сравнение моделей
results = []
for config in configs:
    model, val_loss = build_and_train_model(
        model_type=config["model_type"],
        neurons=config["neurons"],
        layers=config["layers"],
        look_back=look_back,
        x_train=X_train,
        y_train=y_train,
        x_test=X_test,
        y_test=y_test
    )
    predictions = stepwise_prediction(model, data_normalized, look_back)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    results.append({
        "config": config,
        "val_loss": val_loss,
        "predictions": predictions
    })

# Визуализация результатов
plt.figure(figsize=(14, 7))
plt.plot(data, label="Истинные данные", color="blue", linewidth=2)
for result in results:
    config = result["config"]
    label = f"{config['model_type']} | Neurons: {config['neurons']} | Layers: {config['layers']}"
    plt.plot(
        range(look_back, len(data)),
        result["predictions"],
        label=label
    )

plt.title("Сравнение прогнозов моделей")
plt.xlabel("Месяцы")
plt.ylabel("Количество пассажиров")
plt.legend()
plt.show()

# Печать результатов
for result in results:
    config = result["config"]
    print(
        f"Модель: {config['model_type']}, Нейроны: {config['neurons']}, Слои: {config['layers']}, Ошибка (MSE): {result['val_loss']:.5f}")