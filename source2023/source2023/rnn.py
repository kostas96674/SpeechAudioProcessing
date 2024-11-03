import numpy as np
from scipy.signal import medfilt
from sklearn.metrics import accuracy_score
import joblib

# Load features
test_features = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_features.npy')

# Transpose the features to have the correct shape for RNN
test_features = np.transpose(test_features, (0, 2, 1))

# Load labels
combined_labels = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_labels.npy')

# Reshape labels for RNN
combined_labels = combined_labels.reshape(test_features.shape[0], test_features.shape[1])

# Split the data into training and testing sets
X_test = test_features
y_test = combined_labels

# Load the model
model_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/models/rnn_model.joblib'
model = joblib.load(model_path)

# Make predictions and evaluate the model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
y_pred = y_pred.reshape(-1)
y_test = y_test.reshape(-1)

# Apply a median filter with a window length of 5
window_length = 5
y_pred_filtered = medfilt(y_pred, kernel_size=window_length)

# Save the predictions
y_pred_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Predictions/y_pred_filtered_rnn.npy'
np.save(y_pred_path, y_pred_filtered)

accuracy = accuracy_score(y_test, y_pred_filtered)
print(f'RNN (SimpleRNN) Accuracy: {accuracy * 100:.2f}%')
