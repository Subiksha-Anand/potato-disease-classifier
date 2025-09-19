import keras

# Load your .keras model
model = keras.models.load_model(r"C:\Users\subik\OneDrive\Document\PROJECT\potato\saved_model\converted_model.keras")

# Save it in SavedModel format (required by TF Serving)
model.export("potato/saved_model/1")  # creates directory structure expected by TF Serving
