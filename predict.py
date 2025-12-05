import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("drowsiness_model.h5")
test_frames = np.random.rand(1, 10, 64, 64, 1)

pred = model.predict(test_frames)
print("Drowsy" if pred > 0.5 else "Not Drowsy")
