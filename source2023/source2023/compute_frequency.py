import numpy as np
import soundfile as sf

def compute_fundamental_frequency(audio, sr, threshold=0.7):
    """
    Compute the fundamental frequency of a given audio segment using autocorrelation.

    Parameters:
    audio (np.ndarray): Audio data array.
    sr (int): Sample rate of the audio data.
    threshold (float): Threshold for determining a valid fundamental frequency.

    Returns:
    float or None: Fundamental frequency in Hz, or None if no valid frequency is found.
    """
    # Define the range for valid pitch detection (50 Hz to 500 Hz)
    max_lag = int(sr / 50)  # 50 Hz is a common lower bound for human pitch
    min_lag = int(sr / 500)  # 500 Hz is a common upper bound for human pitch

    # Compute the autocorrelation of the audio signal
    autocorr = np.correlate(audio, audio, mode='full')[len(audio)-1:]
    # Normalize the autocorrelation
    autocorr /= autocorr[0]

    # Find the maximum autocorrelation within the defined lag range
    max_autocorr = np.max(autocorr[min_lag:max_lag])
    if max_autocorr < threshold:
        return None

    # Identify the lag corresponding to the maximum autocorrelation
    lag = np.argmax(autocorr[min_lag:max_lag]) + min_lag
    # Calculate the fundamental frequency
    fundamental_freq = sr / lag
    return fundamental_freq

def average_fundamental_frequency(audio_file_path, word_times):
    """
    Compute the average fundamental frequency of words in an audio file.

    Parameters:
    audio_file_path (str): Path to the audio file.
    word_times (list): List of tuples with start and end times (in seconds) of detected words.

    Returns:
    float or None: Average fundamental frequency in Hz, or None if no valid frequencies are found.
    """
    # Load the audio file
    audio, sr = sf.read(audio_file_path)
    fundamental_frequencies = []

    # Iterate through the word times to compute fundamental frequency for each word
    for start, end in word_times:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        word_audio = audio[start_sample:end_sample]
        f0 = compute_fundamental_frequency(word_audio, sr)
        if f0 is not None:
            fundamental_frequencies.append(f0)

    # Compute and return the average fundamental frequency
    if fundamental_frequencies:
        return np.mean(fundamental_frequencies)
    else:
        return None

# Setting the path to the audio file
audio_file_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/test/combined_audio.flac'

# Load the word times
word_times = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Time_limits_of_words/word_times_rnn.npy')

# Compute the average fundamental frequency
average_f0 = average_fundamental_frequency(audio_file_path, word_times)
print(f'Average Fundamental Frequency: {average_f0} Hz')
