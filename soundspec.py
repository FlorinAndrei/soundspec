#!/usr/bin/env python3

from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# common sense limits for frequency
fmin = 10
fmax = 20000

argpar = argparse.ArgumentParser(description="generate spectrogram from sound file")
argpar.add_argument('-b', '--batch', help='batch run, no display, save image to disk', action='store_true')
argpar.add_argument('audiofile', type=str, help='audio file to process')

args = argpar.parse_args()

print('Reading the audio file...')
sf, audio = wavfile.read(args.audiofile)

if sf < fmin:
  print('Sampling frequency too low.')
  sys.exit(1)

# convert to mono
sig = np.mean(audio, axis=1)
# number of points per segment; more points = better frequency resolution
# if equal to sf, then frequency resolution is 1 Hz
npts = int(sf)

print('Calculating FFT...')
f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts)

# FFT at high resolution makes way too many frequencies
# set some lower number of frequencies we would like to keep
# final result will be even smaller after pruning
nf = 1000
# generate an exponential distribution of frequencies
# (as opposed to the linear distribution from FFT)
b = fmin - 1
a = np.log10(fmax - fmin + 1) / (nf - 1)
freqs = np.empty(nf, int)
for i in range(nf):
  freqs[i] = np.power(10, a * i) + b
# list of frequencies, exponentially distributed:
freqs = np.unique(freqs)

# delete frequencies lower than fmin
fnew = f[f >= fmin]
cropsize = f.size - fnew.size
f = fnew
Sxx = np.delete(Sxx, np.s_[0:cropsize], axis=0)

# delete frequencies higher than fmax
fnew = f[f <= fmax]
cropsize = f.size - fnew.size
f = fnew
Sxx = Sxx[:-cropsize, :]

findex = []
# find FFT frequencies closest to calculated exponential frequency distribution
for i in range(freqs.size):
  f_ind = (np.abs(f - freqs[i])).argmin()
  findex.append(f_ind)

# keep only frequencies closest to exponential distribution
# this is usually a massive cropping of the initial FFT data
fnew = []
for i in findex:
  fnew.append(f[i])
f = np.asarray(fnew)
Sxxnew = Sxx[findex, :]
Sxx = Sxxnew

print('Generating the image...')
plt.pcolormesh(t, f, np.log10(Sxx))
plt.ylabel('f [Hz]')
plt.xlabel('t [sec]')
plt.yscale('symlog')
plt.ylim(fmin, fmax)

# TODO: make this depend on fmin / fmax
yt = np.arange(10, 100, 10)
yt = np.concatenate((yt, 10 * yt, 100 * yt, 1000 * yt))
yt = yt[yt <= fmax]
yt = yt.tolist()
plt.yticks(yt)

plt.grid(True)
if args.batch:
  plt.savefig(args.audiofile + '.png', dpi=200)
else:
  plt.show()
