This is a simple machine learning project to detect driver drowsiness from sequences of images.
It uses a Convolutional Neural Network (CNN) for feature extraction and an LSTM model to learn
eye-blinking time patterns and classify whether the driver is Drowsy or Not Drowsy.

## How the Model Works
Frames → CNN → LSTM → Output

## Files
- cnn_lstm.py  → contains CNN + LSTM model
- train.py     → train the model
- predict.py   → test/predict output

## Requirements
TensorFlow
Numpy
OpenCV (optional for real webcam)
