
import numpy as np


def find_main_freq(spec, f):
    
# Amplitude of highest frequency peak

    main_freq_amp = max(abs(spec))        
# Find frequency with highest peak
    pos = np.argwhere(abs(spec) == main_freq_amp)
    main_freq = f[pos]
    return main_freq[0][0], main_freq_amp


def spectral_entropy(spec):
    # Normalized using the sum
    P = spec/sum(spec)
    # Calculating the entropy      
    return -sum(P * np.log2(P))

# zero crossing rate
def zcr(my_array):
    return (((my_array[:-1] * my_array[1:]) < 0).sum())/(my_array.size-1)









