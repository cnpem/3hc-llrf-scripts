#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fn = sys.argv[1]
fs = int(sys.argv[2])

NUM_BITS_ADC = 16
STR_TO_SEARCH_FOR = 'Ch'

data = dict()
with open(fn, 'r') as f:
    while(True):
        index = f.tell()
        line = f.readline()

        if not line:
            break

        if STR_TO_SEARCH_FOR in line:
            index = index + line.find(STR_TO_SEARCH_FOR) + \
                    len(STR_TO_SEARCH_FOR)
            f.seek(index)
            content = f.readline().split(';')[:-1]
            data[int(content[0])] = \
                [(int(sample)/(2**(NUM_BITS_ADC-1))-1)*1000 \
                 for sample in content[1:]]
#plt.figure()
#
#for key in data:
#    t = np.arange(len(data[key]))/fs
#
#    plt.plot(t, data[key], label=f'{key}')
#
#plt.title('Noise Time Decourse')
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude [mV]')
#plt.legend()

#for key in data:
#    plt.figure()
#
#    plt.hist(data[key], bins=50)
#
#    plt.title(f'Histogram of channel {key}')
#    plt.xlabel('Amplitude [mV]')

#plt.figure()
#
#for key in data:
#    D = np.fft.rfft(data[key])
#    f = np.fft.rfftfreq(len(data[key]), 1/fs)
#
#    interest = f >= 0
#    plt.semilogx(f[interest], 20*np.log10(abs(D[interest])), label=f'{key}')
#
#plt.title('Noise FFT')
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Magnitude [dB]')
#plt.legend()
#plt.grid()

fig, ax = plt.subplots(2, 1, sharex=True)

fig.suptitle(
    'SIS8300-KU Noise Analysis\n'
    f'{fn}')

for key in data:
    f, Pxx = signal.welch(data[key], fs, nperseg=len(data[key])//100)
    Pxx_rms = np.sqrt(np.cumsum(Pxx)*(f[1] - f[0]))

    interest = f > 0
    f, Pxx, Pxx_rms = f[interest], Pxx[interest], Pxx_rms[interest]

    ax[0].loglog(f, Pxx)

    ax[1].semilogx(f, Pxx_rms, label=f'{key}')

ax[0].set_ylabel("NSD [mV²/Hz]")
ax[0].grid(True, which='both')

ax[1].set_ylabel("Cumulative RMS noise [mV]")
ax[1].set_xlabel("Frequency [Hz]")
ax[1].grid(True, which='both')
ax[1].legend()

plt.tight_layout()
plt.show()
