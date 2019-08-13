# soundspec
spectrum visualizer for audio files

## Output example
Here's an output sample from running the app on a WAV file:

```
$ ./soundspec.py joker.wav 
Reading the audio file...
Calculating FFT...
Generating the image...
```

This is the image that was generated. The horizontal axis is time (in seconds). The vertical axis is frequency (in Hz). The colors represent sound intensities - blue is quiet, yellow is loud.

![joker.png](joker.png)

It's the spectrum of the song 'Why So Serious?' by Hans Zimmer and James Newton Howard, from the movie 'The Dark Knight'.

The song is semi-famous among "audiophile" enthusiasts, where it is considered a good test of bass response for audio systems. The portion of interest begins shortly after 200 sec (3 min 24 sec, more or less). You can see a lot of energy is focused between 30 Hz and 40 Hz. While pretty low, these are not actually extremely low frequencies. If your speakers or headphones can play those notes, they are alright.

## Technical explanation
The app will take any WAV file as input. It performs the fast Fourier transform (FFT) to generate the spectrum of the whole file. The spectrum is then displayed.

Only WAV files can be used as input for now. Ingesting arbitrary file formats would require external dependencies such as ffmpeg, etc. Many apps can be used to convert your files to WAV - a good example is VLC.

The app has no GUI, it is strictly intended to be launched from the command line.

The current version has not been tested much. If it crashes, let me know - use the Issues link at the top of this page and open a case.
