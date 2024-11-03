import librosa
import soundfile as sf

def split_audio_clip(path, output_dir):
    """
    Split an audio file into 4-second clips and save them as 24-bit FLAC files.

    Parameters:
    path (str): Path to the input audio file.
    output_dir (str): Directory where the output clips will be saved.
    """
    # Get the sample rate of the audio file
    sr = librosa.get_samplerate(path)
    # Get the duration of the audio file in seconds
    sec = int(librosa.get_duration(filename=path, sr=sr))
    
    # Loop through the audio file in 4-second intervals
    for i in range(0, sec, 4):
        # Load a 4-second segment of the audio file starting at 'i' seconds
        y, sr = librosa.load(path, offset=i, sr=sr, duration=4)
        # Write the audio segment to a 24-bit FLAC file
        sf.write(f'{output_dir}/noise-{900+i}.flac', data=y, samplerate=sr, format='flac', subtype='PCM_24')

# Example usage
file_path = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-background/Recording_15.flac'
output_dir = 'C:/Users/kosta/Desktop/SpeechAudioProcessing/auxilary2023/LibriSpeech/dev-background'

# Split the audio file into clips and save them
split_audio_clip(file_path, output_dir)

