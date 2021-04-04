import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
#from scipy.io.wavfile import write

SAMPLE_RATE = 44100
AMPLITUDE = 4096

def get_wave(frequency,duration):
    t = np.linspace(0,duration,int(SAMPLE_RATE*duration)) #Equally distributed time intervals to mimic a wave
    wave = AMPLITUDE*np.sin(2*np.pi*frequency*t)
    return wave
  
#a_wave = get_wave(440,1)
#print(len(a_wave))
#type(a_wave)
#plt.plot(a_wave[0:int(44100/440)])
#plt.xlabel('time')
#plt.ylabel('Amplitude')
#plt.show()


##################### SINGLE OCTAVE KEY NOTES OF PIANO #####################
#def get_piano_notes():
#    octave = ['C','c','D','d','E','F','f','G','g','A','a','B']
#    base_freq = 523.25 #C5 note frequency
    
#    note_freqs = {octave[i]: base_freq*pow(2,(i/12)) for i in range(len(octave))}
    
#    note_freqs[''] = 0.0 #silent note
    
#    return note_freqs

##################### 88 KEY NOTES OF PIANO #####################
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
#c4_freq = note_freqs['C4']
#pprint(note_freqs)

#notes = 'c+E+c+c+E+c+c+E+c+d+c+B' #Shape of You Melody

## WRITE SONG

#ef get_song_data(music_notes):
#   note_freqs = get_piano_notes()
#   song = [get_wave(note_freqs[note]) for note in music_notes.split('+')]
#   song = np.concatenate(song)
#   return song
  
#ata = get_song_data(notes)

#rite('shape_of_you_melody.wav', frequency, data.astype(np.int16))

def get_adsr_weights(frequency, duration, length, decay, sustain_level,sample_rate=44100):

    assert abs(sum(length)-1) < 1e-8
    assert len(length) ==len(decay) == 4
    
    intervals = int(duration*frequency)
    len_A = np.maximum(int(intervals*length[0]),1)
    len_D = np.maximum(int(intervals*length[1]),1)
    len_S = np.maximum(int(intervals*length[2]),1)
    len_R = np.maximum(int(intervals*length[3]),1)
    
    decay_A = decay[0]
    decay_D = decay[1]
    decay_S = decay[2]
    decay_R = decay[3]
    
    A = 1/np.array([(1-decay_A)**n for n in range(len_A)])
    A = A/np.nanmax(A)
    D = np.array([(1-decay_D)**n for n in range(len_D)])
    D = D*(1-sustain_level)+sustain_level
    S = np.array([(1-decay_S)**n for n in range(len_S)])
    S = S*sustain_level
    R = np.array([(1-decay_R)**n for n in range(len_R)])
    R = R*S[-1]
    
    weights = np.concatenate((A,D,S,R))
    smoothing = np.array([0.1*(1-0.1)**n for n in range(5)])
    smoothing = smoothing/np.nansum(smoothing)
    weights = np.convolve(weights, smoothing, mode='same')
    
    weights = np.repeat(weights, int(sample_rate*duration/intervals))
    tail = int(sample_rate*duration-weights.shape[0])
    if tail > 0:
        weights = np.concatenate((weights, weights[-1]-weights[-1]/tail*np.arange(tail)))
    return weights
