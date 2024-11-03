import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from load import load_audio_files, extract_mel_spectrogram
from tensorflow.keras.layers import TimeDistributed
import joblib

# Load foreground and background audio data
foreground_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-forground')
background_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-background')

# Extract features
foreground_features = extract_mel_spectrogram(foreground_audio)
background_features = extract_mel_spectrogram(background_audio)

# Transpose the features to have the correct shape for RNN
foreground_features = np.transpose(foreground_features, (0, 2, 1))
background_features = np.transpose(background_features, (0, 2, 1))

# Create labels
foreground_labels = np.ones(foreground_features.shape[1]*len(foreground_features))
background_labels = np.zeros(background_features.shape[1]*len(background_features))

# Combine features and labels
features = np.concatenate((foreground_features, background_features), axis=0)
labels = np.concatenate((foreground_labels, background_labels), axis=0)

# Reshape labels for RNN
labels = labels.reshape(features.shape[0], features.shape[1])

# Data for training
X_train = features
y_train = labels

# Initialize and train the SimpleRNN model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(SimpleRNN(50, return_sequences=True))
model.add(SimpleRNN(50, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=2)

# Save the model to a file
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/rnn_model.joblib'
joblib.dump(model, model_path)


