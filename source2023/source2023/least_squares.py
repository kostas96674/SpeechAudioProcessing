import joblib
import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score

#Load fetures and labels
test_features = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_features.npy')
combined_labels = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_labels.npy')

# Reshape the features and labels
test_features = test_features.reshape(-1,test_features.shape[1])
combined_labels = combined_labels.reshape(-1)

# Initialize the test set
X_test = test_features
y_test = combined_labels

# Load the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/linear_regression_model.joblib'
lin_reg = joblib.load(model_path)

# Make predictions
y_pred = lin_reg.predict(X_test)

# Apply a threshold to get binary class labels
threshold = 0.5
y_pred_binary = np.where(y_pred >= threshold, 1, 0)

# Apply a median filter with a window length of 5
window_length = 5
y_pred_filtered = medfilt(y_pred_binary, kernel_size=window_length)

# Save the predictions
y_pred_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Predictions/y_pred_filtered_least_squares.npy'
np.save(y_pred_path, y_pred_filtered)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred_filtered)
print(f'Least Squares (Linear Regression) Accuracy: {accuracy * 100:.2f}%')
