import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.io.wavfile import write

frequency = 44100

def get_wave(freq,duration = 0.5):
    amplitude = 4096
    t = np.linspace(0,duration,int(frequency*duration))
    wave = amplitude*np.sin(2*np.pi*freq*t)
    
    return wave
  
  a_wave = get_wave(440,1)
  
  #print(len(a_wave))
  #type(a_wave)
  
plt.plot(a_wave[0:int(44100/440)])
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()


###########

def get_piano_notes():
    octave = ['C','c','D','d','E','F','f','G','g','A','a','B']
    base_freq = 523.25 #C5 note frequency
    
    note_freqs = {octave[i]: base_freq*pow(2,(i/12)) for i in range(len(octave))}
    
    note_freqs[''] = 0.0 #silent note
    
    return note_freqs

def get_piano_notes():
    octave = ['C','c','D','d','E','F','f','G','g','A','a','B']
    base_freq = 440 #A4 frequency in Hz

    keys = np.array([x+str(y) for y in range(0,9) for x in octave])

    start_index = np.where(keys == 'A0')[0][0]
    end_index = np.where(keys == 'C8')[0][0]
    keys = keys[start_index:end_index+1]

    note_freqs = dict(zip(keys,[2**((n+1-49)/12)*base_freq for n in range(len(keys))]))
    note_freqs[''] = 0.0

    return note_freqs
  
note_freqs = get_piano_notes()

pprint(note_freqs)

notes = 'c+E+c+c+E+c+c+E+c+d+c+B' #Shape of You Melody
#notes = 'c++E++c++c++E++c++c++E++c++d++c++B'

## WRITE SONG

def get_song_data(music_notes):
    note_freqs = get_piano_notes()
    song = [get_wave(note_freqs[note]) for note in music_notes.split('+')]
    song = np.concatenate(song)
    return song
  
data = get_song_data(notes)

write('shape_of_you_melody.wav', frequency, data.astype(np.int16))
