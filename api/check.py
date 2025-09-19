import tensorflow as tf

model_path = r"C:\Users\subik\OneDrive\Document\PROJECT\potato\saved_model\converted_model.keras" # Change to your model's path
model = tf.keras.models.load_model(model_path)

if model is None:
    print("Error: Model failed to load!")
else:
    result = model.predict(your_input)
