import numpy as np
from sklearn.neural_network import MLPClassifier
from load import load_audio_files, extract_mel_spectrogram 
import joblib

# Load foreground and background audio data
foreground_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-forground')
background_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-background')

# Extract features
foreground_features = extract_mel_spectrogram(foreground_audio)
background_features = extract_mel_spectrogram(background_audio)

# Create labels
foreground_labels = np.ones(foreground_features.shape[2]*len(foreground_features))
background_labels = np.zeros(background_features.shape[2]*len(background_features))

# Combine features and labels
X = np.concatenate((foreground_features, background_features), axis=0)
y = np.concatenate((foreground_labels, background_labels), axis=0)

# Reshape the features and lables for MLP
X = X.reshape(-1,X.shape[1])
y = y.reshape(-1)

# Data for training
X_train = X
y_train = y

# Initialize and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=100, random_state=42)
mlp.fit(X_train, y_train)

#Save the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/mlp_model.joblib'
joblib.dump(mlp, model_path)


