import numpy as np
from load import load_audio_files, extract_mel_spectrogram , extract_single_mel_spectrogram
import soundfile as sf
import random

# Function to save combined audio
def save_audio(audio_data, sample_rate, output_path):
    with sf.SoundFile(output_path, mode='w', samplerate=sample_rate, channels=1, format='FLAC') as file:
        file.write(audio_data)

# Load foreground and background audio data
foreground_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-forground')
background_audio = load_audio_files('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-background')

# Extract features
foreground_features = extract_mel_spectrogram(foreground_audio)
background_features = extract_mel_spectrogram(background_audio)

# Create labels
foreground_labels = np.ones(foreground_features.shape[2] * len(foreground_features))
background_labels = np.zeros(background_features.shape[2] * len(background_features))

# Combine features and labels
X = np.concatenate((foreground_features, background_features), axis=0)
y = np.concatenate((foreground_labels, background_labels), axis=0)

# Select a subset of foreground and background audio for mashing
selected_foreground_audio = random.sample(foreground_audio, 5)
selected_background_audio = random.sample(background_audio, 5)

# Combine selected audio into a new audio file
combined_audio = []
combined_labels = []
sr = 22050 

for fg_audio, bg_audio in zip(selected_foreground_audio, selected_background_audio):
    fg_features = extract_single_mel_spectrogram(fg_audio)
    bg_features = extract_single_mel_spectrogram(bg_audio)
    combined_audio.append(fg_audio)
    combined_labels.append(np.ones(fg_features.shape[1]))
    combined_audio.append(bg_audio)
    combined_labels.append(np.zeros(bg_features.shape[1]))

combined_audio = np.array(combined_audio)
combined_labels = np.array(combined_labels)

# Save the combined audio file
output_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/test/combined_audio.flac'
save_audio(combined_audio.flatten(), sr, output_path)

# Extract features from combined audio
combined_features = extract_single_mel_spectrogram(combined_audio)

# Save the combined features
features_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_features.npy'
np.save(features_path, combined_features)

# Save the combined labels
labels_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/combined_labels.npy'
np.save(labels_path, combined_labels)
