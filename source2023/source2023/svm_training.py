import numpy as np
from sklearn.svm import LinearSVC
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

# Reshape the features and lables for SVM
X = X.reshape(-1,X.shape[1])
y = y.reshape(-1)

# Data for training
X_train = X
y_train = y

# Initialize and train the SVM model
svm_model = LinearSVC(random_state=42, max_iter=1000,dual=False)
svm_model.fit(X_train, y_train)

#Save the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/svm_model.joblib'
joblib.dump(svm_model, model_path)


