from tensorflow.keras.models import load_model

model = load_model("best_model (1).h5", compile=False)
model.summary()
