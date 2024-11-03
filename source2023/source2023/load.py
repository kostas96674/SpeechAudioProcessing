import os
import librosa
import numpy as np

def load_audio_files(directory):
    """
    Load audio files from a given directory.

    Parameters:
    directory (str): Path to the directory containing audio files.

    Returns:
    list: List of loaded audio data.
    """
    audio_files = []
    # Traverse the directory tree and load audio files
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Load the audio file with a fixed duration of 3 seconds
            audio, _ = librosa.load(file_path, duration=3)
            audio_files.append(audio)
    return audio_files

def extract_mel_spectrogram(audio_data, n_mels=96, hop_length=512, n_fft=1024):
    """
    Extract mel spectrogram features from audio data.

    Parameters:
    audio_data (list): List of audio data arrays.
    n_mels (int): Number of mel bands to generate.
    hop_length (int): Number of samples between successive frames.
    n_fft (int): Length of the FFT window.

    Returns:
    np.ndarray: Array of mel spectrogram features.
    """
    features = []
    for audio in audio_data:
        if len(audio) == 0:
            continue
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
        # Convert the mel spectrogram to log scale (dB)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        features.append(log_mel_spectrogram)
    return np.array(features)

def extract_single_mel_spectrogram(audio_data, n_mels=96, hop_length=512, n_fft=1024):
    """
    Extract a single mel spectrogram feature from an audio data array.

    Parameters:
    audio_data (np.ndarray): Audio data array.
    n_mels (int): Number of mel bands to generate.
    hop_length (int): Number of samples between successive frames.
    n_fft (int): Length of the FFT window.

    Returns:
    np.ndarray: Log-scaled mel spectrogram.
    """
    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    # Convert the mel spectrogram to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram
