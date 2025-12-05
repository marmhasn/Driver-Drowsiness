import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_lstm_model():
    model = models.Sequential()
    model.add(layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu'), input_shape=(10, 64, 64, 1)))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Conv2D(64, (3,3), activation='relu')))
    model.add(layers.TimeDistributed(layers.MaxPooling2D((2,2))))
    model.add(layers.TimeDistributed(layers.Flatten()))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

