import numpy as np
import tensorflow as tf
import pandas as pd

# Load the trained model (adjust path if saved differently)
model = tf.keras.models.load_model("nn_model.h5")  # Save your model first!

# Function to convert address to vector (same as training)
def address_to_vector(address):
    byte_data = bytes.fromhex(address.replace("0x", ""))
    return list(byte_data)

# Max value for denormalization (2**256 - 1)
y_max = float(2**256 - 1)

# Sample function to predict integer from address
def predict_integer(wallet_address):
    # Preprocess input
    vector = address_to_vector(wallet_address)
    X_input = np.array([vector]) / 255.0  # Normalize like training data
    
    # Predict (returns normalized 0-1)
    pred_normalized = model.predict(X_input, verbose=0).flatten()[0]
    
    # Denormalize to 2**256 scale
    pred_integer = pred_normalized * y_max
    return int(pred_integer)  # Convert to integer

# Example usage
if __name__ == "__main__":
    # Test with a sample address (replace with your own)
    sample_address = "0x5fC2f4D1e9f1a68404f9605A19c566D9eC8bA9Fc"
    predicted_int = predict_integer(sample_address)
    print(f"Wallet Address: {sample_address}")
    print(f"Predicted Integer: {predicted_int}")

    # Interactive input
    user_address = input("Enter a wallet address (e.g., 0x...): ")
    predicted_int = predict_integer(user_address)
    print(f"Predicted Integer for {user_address}: {predicted_int}")