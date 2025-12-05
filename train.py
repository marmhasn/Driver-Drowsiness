from cnn_lstm import build_cnn_lstm_model
import numpy as np

X_train = np.random.rand(20, 10, 64, 64, 1)
y_train = np.random.randint(0, 2, 20)

model = build_cnn_lstm_model()
model.fit(X_train, y_train, epochs=5, batch_size=2)
model.save("drowsiness_model.h5")
