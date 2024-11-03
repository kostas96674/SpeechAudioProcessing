import numpy as np

def scan_words(y_pred_filtered, hop_length=512, sr=22050):
    """
    Scan through the filtered predictions to identify word boundaries.

    Parameters:
    y_pred_filtered (np.ndarray): Array of filtered predictions (1 for word, 0 for silence).
    hop_length (int): Number of samples between successive frames (default: 512).
    sr (int): Sample rate of the audio (default: 22050).

    Returns:
    list: List of tuples with start and end times (in seconds) of detected words.
    """
    words = []
    i = 0
    # Scan through the predictions to find segments marked as words
    while i < len(y_pred_filtered):
        # Skip the silence parts
        while i < len(y_pred_filtered) and y_pred_filtered[i] == 0:
            i += 1
        # Mark the start of a word
        if i < len(y_pred_filtered) and y_pred_filtered[i] == 1:
            start = i
            # Find the end of the word
            while i < len(y_pred_filtered) and y_pred_filtered[i] == 1:
                i += 1
            end = i - 1
            if start != end:
                words.append((start, end))
    # Convert frame boundaries to seconds
    word_times = [(start * hop_length / sr, end * hop_length / sr) for start, end in words]
    return word_times

# Load the predictions
y_pred_filtered = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Predictions/y_pred_filtered_rnn.npy')

# Scan the words
word_times = scan_words(y_pred_filtered)
print("Word times (in seconds):", word_times)

# Save the word times
filepath = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Time_limits_of_words/word_times_rnn.npy'
np.save(filepath, word_times)
