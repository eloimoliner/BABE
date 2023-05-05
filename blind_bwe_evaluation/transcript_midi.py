
import os
import glob
from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

path="original"
path_midi="midi_bd/original_processed"
files=glob.glob(os.path.join(path,"*.wav"))

sample_rate=22050

for f in files:
        print(f)
        out_file=os.path.join(path_midi, os.path.basename(f))
        if os.path.exists(out_file):
            continue

        # Load audio
        (audio, _) = load_audio(f, sr=sample_rate, mono=True)

        # Transcriptor
        transcriptor = PianoTranscription(device='cuda')    # 'cuda' | 'cpu'

        # Transcribe and write out to MIDI file
        transcribed_dict = transcriptor.transcribe(audio, out_file+'.mid')
