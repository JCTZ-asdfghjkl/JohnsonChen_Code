import pretty_midi
import numpy as np

# 1) Load the MIDI file
midi = pretty_midi.PrettyMIDI(r"../../archive/deap-dataset/audio_stimuli_MIDI/exp_id_1.mid")
print(midi)

# 2) Convert to a piano-roll at 100 frames-per-second
#    Returns a (128, T) array of velocities
piano_roll = midi.get_piano_roll(fs=100)
print(piano_roll)

# 3) (Optional) Convert to a boolean on/off array
on_off = piano_roll > 0

print("Piano-roll shape:", piano_roll.shape)
# e.g. (128 pitches, 5000 time bins for a 50 s file at 100 Hz)
