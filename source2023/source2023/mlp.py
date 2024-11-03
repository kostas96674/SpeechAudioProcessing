from scipy.signal import medfilt
from sklearn.metrics import accuracy_score
import numpy as np
import joblib

# Load the features and labels
test_features = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_features.npy')
combined_labels = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_labels.npy')


# Reshape the features and labels for MLP
test_features = test_features.reshape(-1,test_features.shape[1])
combined_labels = combined_labels.reshape(-1)

# Data for testing
X_test = test_features
y_test = combined_labels

# Load the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/mlp_model.joblib'
mlp = joblib.load(model_path)

# Make predictions and evaluate the model
y_pred = mlp.predict(X_test)

# Apply a median filter with a window length of 7
window_length = 7
y_pred_filtered = medfilt(y_pred, kernel_size=window_length)

# Save the predictions
y_pred_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Predictions/y_pred_filtered_mlp.npy'
np.save(y_pred_path, y_pred_filtered)

accuracy = accuracy_score(y_test, y_pred_filtered)
print(f'MLP Accuracy: {accuracy * 100:.2f}%')
