import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasource = r"C:\Users\msiso\Desktop\scrappy-seacrh\pkey\eth.csv"
# datasource = r"C:\Users\msiso\Desktop\scrappy-seacrh\pkey\trx.csv"

# Load and preprocess
df = pd.read_csv(datasource)
def address_to_vector(address):
    byte_data = bytes.fromhex(address.replace("0x", ""))
    return list(byte_data)

df["address_vector"] = df["wallet_address"].apply(address_to_vector)
X = np.array(df["address_vector"].tolist()) / 255.0  # Normalize to 0-1
y = df["integer"].values

y = np.array([float(yi) for yi in y], dtype=np.float64)
print(f"Min: {y.min()}, Max: {y.max()}, Mean: {y.mean()}")

y_max = float(2**256 - 1)
y = y / y_max

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Deeper model with dropout to prevent overfitting
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),  # Reduce overfitting
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])

# Train longer
history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

# Predict and evaluate
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE with vector encoding: {mae:.2f}")

for i in range(5000, 5010):
    print(f"Predicted: {y_pred[i]:.0f}, Actual: {y_test[i]}")

model.save("nn_model.h5")