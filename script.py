import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
#from scipy.io.wavfile import write

def get_wave(frequency, duration, sample_rate = 44100, amplitude = 4096):
    '''
    Returns an array of sine wave of a particular frequency.
    '''
    t = np.linspace(0,duration,int(sample_rate*duration)) #Equally distributed time intervals to mimic a wave
    wave = amplitude*np.sin(2*np.pi*frequency*t)
    return wave

def get_piano_notes():
    '''
    Returns dict of frequencies of a 88 key standard piano.
    Math equation for frequency calculation: 2^((n-49)/12)*440Hz where n is key number from A4 (440 Hz)
    '''

    octave = ['C','c','D','d','E','F','f','G','g','A','a','B']
    base_freq = 440 #A4 frequency in Hz

    keys = np.array([x+str(y) for y in range(0,9) for x in octave])

    start_index = np.where(keys == 'A0')[0][0]
    end_index = np.where(keys == 'C8')[0][0]
    keys = keys[start_index:end_index+1]

    note_freqs = dict(zip(keys,[2**((n-49)/12)*base_freq for n in range(1,len(keys)+1)]))
    note_freqs[''] = 0.0

    return note_freqs
  
def apply_overtones(frequency, duration, factor = [0.67, 0.25, 0.02, 0.02, 0.04], sample_rate = 44100, amplitude = 4096):
    '''
    Returns a note with overtones applied.
    Default is 4 overtones on the fundamental note.
    '''
    
    frequencies = np.minimum(np.array([frequency*(x) for x in range(1,len(factor)+1)]), sample_rate//3)
    amplitudes = np.array([amplitude*x for x in factor])
    
    fundamental = get_wave(frequencies[0], duration, sample_rate, amplitudes[0])
    for i in range(1, len(factor)):
        overtone = get_wave(frequencies[i], duration, sample_rate, amplitudes[i])
        fundamental += overtone
    return fundamental

def get_adsr_weights(frequency, duration, length, decay, sustain_level, sample_rate=44100):
    
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

def apply_pedal(note_durations, bar_duration):
    
    new_durations = []
    start = 0

    while True:
        # Count total duration from end of last bar
        cum_value = np.cumsum(np.array(note_durations[start:]))
        # Find end of this bar
        end = np.where(cum_value == bar_duration)[0][0]
        if end == 0: # If the note takes up the whole bar
            new_durations += [note_durations[start]]
        else:
            this_bar = np.array(note_durations[start:start+end+1])
            # New value of note is the remainder of bar = (total duration of bar) - (cumulative duration thus far)
            new_durations += [bar_duration-np.sum(this_bar[:i]) for i in range(len(this_bar))]
        start += end+1
        if start == len(note_durations):
            break
    return new_durations

def get_song_data(music_notes, note_durations, bar_duration, factor, length, decay, sustain_level, sample_rate = 44100, amplitude = 4096):
    note_freqs = get_piano_notes()
    frequencies = [note_freqs[note] for note in music_notes]
    new_durations = apply_pedal(note_durations, bar_duration)
    duration = int(sum(note_durations)*sample_rate)
    end_idx = np.cumsum(np.array(note_durations)*sample_rate).astype(int)
    start_idx = np.concatenate(([0], end_idx[:-1]))
    end_idx = np.array([start_idx[i]+new_durations[i]*sample_rate for i in range(len(new_durations))]).astype(int)
    
    song = np.zeros((duration,))
    
    for i in range(len(music_notes)):
        this_note = apply_overtones(frequencies[i], new_durations[i], factor)
        weights = get_adsr_weights(frequencies[i], new_durations[i], length, decay, sustain_level)
        song[start_idx[i]:end_idx[i]] += this_note*weights

    song = song*(amplitude/np.max(song))
    return song
