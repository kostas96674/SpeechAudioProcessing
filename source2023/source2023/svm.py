from scipy.signal import medfilt
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

#Load test features
test_features = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_features.npy')

# Load test labels
combined_labels = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_labels.npy')

# Reshape the features and labels for SVM
test_features = test_features.reshape(-1,test_features.shape[1])
combined_labels = combined_labels.reshape(-1)

# Split the data into training and testing sets
X_test = test_features
y_test = combined_labels

#Load the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/svm_model.joblib'
svm_model = joblib.load(model_path)

# Make predictions and evaluate the model
y_pred = svm_model.predict(X_test)

# Example: Apply a median filter with a window length of 5
window_length = 5
y_pred_filtered = medfilt(y_pred, kernel_size=window_length)

# Save the predictions
y_pred_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Predictions/y_pred_filtered_svm.npy'
np.save(y_pred_path, y_pred_filtered)

accuracy = accuracy_score(y_test, y_pred_filtered)
print(f'SVM Accuracy: {accuracy * 100:.2f}%')




