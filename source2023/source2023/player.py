import sounddevice as sd
import soundfile as sf
import numpy as np

def play_audio_segment(audio_segment, sample_rate):
    """
    Play a given audio segment.

    Parameters:
    audio_segment (np.ndarray): Audio data to be played.
    sample_rate (int): Sample rate of the audio data.
    """
    sd.play(audio_segment, sample_rate)
    sd.wait()

def play_words(audio_file_path, word_times):
    """
    Play the audio segments corresponding to the detected words.

    Parameters:
    audio_file_path (str): Path to the audio file.
    word_times (list): List of tuples with start and end times (in seconds) of detected words.
    """
    # Load the audio file
    audio, sr = sf.read(audio_file_path)
    for start, end in word_times:
        # Convert start and end times to sample indices
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        # Extract the audio segment for the word
        word_audio = audio[start_sample:end_sample]
        # Play the audio segment
        play_audio_segment(word_audio, sr)

# Path to the audio file
audio_file_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/test/combined_audio.flac'
# Load the word times
word_times = np.load('C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/Time_limits_of_words/word_times.npy')

# Play the detected words
play_words(audio_file_path, word_times)
