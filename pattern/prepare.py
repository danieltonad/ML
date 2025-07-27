import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# datasource = r"C:\Users\msiso\Desktop\scrappy-seacrh\pkey\trx.csv"
datasource = r"C:\Users\msiso\Desktop\scrappy-seacrh\pkey\eth.csv"

# Load the data
df = pd.read_csv(datasource)
# print(df)

# Encode wallet address as numerical features
# We'll convert the hex string to an integer (simplified)
def address_to_vector(address):
    hex_str = address.replace("0x", "")
    return [int(c, 16) for c in hex_str]


# Features (X) and target (y)
df["address_vector"] = df["wallet_address"].apply(address_to_vector)
X = np.array(df["address_vector"].tolist())  # Shape: (1000, 10)
y = df["integer"].values

# print(X[:5], y[:5])



# Train a model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# Train the model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# # Predict on test set
# y_pred = model.predict(X_test)

# mae = mean_absolute_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# print(f"MAE with vector encoding: {mae:.2f}")
# for i in range(5):
#     print(f"Predicted: {y_pred[i]:.0f}, Actual: {y_test[i]}")
# ================================================================================

# Tensor 
# ==============================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred.flatten())
print(f"MAE with vector encoding: {mae:.2f}")

for i in range(5):
    print(f"Predicted: {y_pred[i][0]:.0f}, Actual: {y_test[i]}")

# ==============================================================================

# Modify predictions to give a range
# def predict_range(model, address_num, tolerance=5):
#     pred = model.predict([[address_num]])[0]
#     return (max(0, int(pred - tolerance)), int(pred + tolerance))

# # Test it
# test_range = predict_range(model, test_address_num)
# print(f"Predicted range for {test_address}: {test_range}")


# SAVE MODEL 

import joblib

# Save the trained model to a file
joblib.dump(model, "random_forest_model.pkl")

# Load it later
# loaded_model = joblib.load("random_forest_model.pkl")

# # Use it to predict
# test_address_num = 5962486840  # Example from your data
# prediction = loaded_model.predict([[test_address_num]])
# print(f"Loaded model prediction: {prediction[0]:.0f}")