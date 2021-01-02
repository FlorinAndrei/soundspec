# soundspec
spectrum visualizer for audio files; uses the Fourier transform

Requires Python 3 and a few typical number crunching modules. Install the modules and run the `soundspec.py` script as is. It should be self-explanatory. It works on these type of files: WAV, FLAC, MP3, MP4, MKV, AVI, OGG, WEBM. It uses ffmpeg to convert files other than WAV to WAV. If your file type is not supported convert it to WAV first with ffmpeg or VLC.

There's an .sh file and a .bat file for batch processing - analyzing many files with one command. Those require ffmpeg, and can analyze any type of audio file - MP3, M4A, FLAC, etc, if it's supported by ffmpeg, it will probably work.

The core part of the script is the Fourier transform done via `scipy.signal.spectrogram()`. I have not done any automation around the file parameters, but it should work with typical audio files containing music. Open an issue on GitHub if something doesn't work as it should, or hack the code and send me a pull request.

Currently, the frequency resolution is about 1 Hz (in the bottom part of the spectrum), and time resolution is about 1 second. Details finer than those limits will seem smeared. Resolution could be increased, but at the cost of increasing the run time of the app.

## For Windows 10 users (Mac version will follow soon, I hope)

If you don't want to mess with Python scripting, there's an .exe for Windows you could download. Click the [releases](https://github.com/FlorinAndrei/soundspec/releases) link at the top of the page and download the most recent version. Extract the zip archive in some convenient folder - I prefer something like `C:\opt\soundspec` but that's up to you.

Within the archive you'll find two executables: `soundspec.exe` and `soundspec-batch.bat`.

### Analyzing one file at a time

`soundspec.exe` is for making the spectrum of a one or more files. It only works on WAV files. Example:

```
soundspec.exe "C:\Users\darkstar\Music\latest hit.wav"
```

You must run it from the Command Prompt. Open cmd.exe, navigate to the place where you've extracted the zip archive, and run the app as shown above. Provide the full path to the music file, and use double quotes.

There's no GUI initially, sorry; a simple GUI will open to show you the spectrum image, but that's all. You can save the image from the GUI, or zoom in, scroll, etc.

It should not take more than a couple seconds to analyze a file, in most cases.

### Analyzing many files at once

First you need to install [ffmpeg](https://ffmpeg.org/). The Windows version is currently [on this page](https://ffmpeg.zeranoe.com/builds/). I suggest you avoid the nightly builds, and stick to the release builds instead.

Download and install it in a place such as `C:\opt\ffmpeg`. Then [add the bin directory to the PATH](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/) - it should look like `C:\opt\ffmpeg\bin`. Then if you open the Command Prompt and type "ffmpeg", it should run it no matter in which folder you are currently. That's a requirement.

Download the music files you want to analyze, and place them in any directory. Open a Command Prompt, go to the soundspec folder, and run the batch script with the full path to the folder containing your music files. Example:

```
soundspec-batch.bat "C:\Users\darkstar\Music\Bananas album"
```

It will take a few seconds to analyze each song. Next to the songs you'll find PNG files with the spectral analysis for each song. If you give it thousands of files, it may take hours to complete; it will not overload your system since only one song is processed at any given time.

You may also use the batchmode with option -b:

```
soundspec.py -b "C:\Users\darkstar\Music\Bananas album\*.wav"
```

The batchmode uses all available cores to speed up processing which leads to 100% CPU load and may even bring your system into heavy swapping if memory is not sufficient. In this case use option -c to reduce the number of cores used (see below in chapter Batchmode Options, as well for other options).

There's no GUI at all in batch mode. Almost any audio file format is supported in this mode, including MP3, M4A, FLAC, etc. Feel free to mix and match different file types.

## For Ubuntu users

Install any missing `python 3` libraries, e.g. `scipy` and `numpy`, as follows:

```
sudo apt install python3-scipy
sudo apt install python3-numpy
```

Install `ffmpeg` (without it only WAV files are supported):

```
sudo apt install ffmpeg
```

### Analyzing files

`soundspec.py` is for making spectrograms of audio files. It works on many different file types if `ffmpeg` is installed. Examples:

```
./soundspec.py "/home/darkstar/Music/latest hit.wav"
./soundspec.py -b "/home/darkstar/Music/*.mp3"
./soundspec.py -b -c4 "/home/darkstar/Music/"
```

You must run it from a unix shell:
- Open a terminal application, navigate to the place where you've extracted the zip archive, and run the app as shown above. 
- Provide the full path to audio files or to folders containing audio files. If you provide a folder `soundspec` processes all audio files in this folders and all its nested subfolders.
- Use double quotes if the path names contain any blanks.

## Common options

### -d <debug_level>
This option sets the debug level (a number greater than 0). The higher the number the more debug messages are printed and the slower the app performs.

### -w <window>
This options sets the window (a string) for the fourier transformation. The default is 'blackmanharris'. See this page for all supported window types:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.get_window.html#scipy.signal.get_window

## Batchmode options

### -c <num_cores>
In batchmode the app uses all available cores to speed up processing which leads to 100% CPU load and may even bring your system into heavy swapping if memory is not sufficient. If this happens you may specifiy how many cores the app may use with option -c <num_cores>:

```
soundspec.py -b -c 2 "C:\Users\darkstar\Music\Bananas album\*.wav"
```

### -p <ppi> 
This option allows to set the ppi of the graphic saved on disk. The default is 200 which leads to a file size of approximately more than 1 MB. Reduce the number to reduce the file size:

```
soundspec.py -b -p 150 "C:\Users\darkstar\Music\Bananas album\*.wav"
```


## Output example
Here's an output sample from running the app on a WAV file:

```
$ ./soundspec.py joker.wav 
joker.wav: reading ...
joker.wav: processing ...
joker.wav: calculating FFT ...
joker.wav: generating the image ...
```

This is the image that was generated. The horizontal axis is time (in seconds). The vertical axis is frequency (in Hz). The colors represent sound intensities - blue is quiet, yellow is loud.

![joker.png](joker.png)

It's the spectrum of the song ['Why So Serious?'](https://www.youtube.com/watch?v=1zyhQjJ5UgY) by Hans Zimmer and James Newton Howard, from the movie 'The Dark Knight'.

The song is semi-famous among "audiophile" enthusiasts, where it is considered a good test of bass response for audio systems. The portion of interest begins shortly after 200 sec (3 min 24 sec, more or less). You can see a lot of energy is focused between 30 Hz and 40 Hz. While pretty low, these are not actually extremely low frequencies - there are songs out there with deeper bass, closer to 20 Hz in some cases. But if your speakers or headphones can play those notes, they are alright.

## Notes

Discussion thread about this app on the Audio Science Review forum:

https://www.audiosciencereview.com/forum/index.php?threads/simple-app-to-visualize-the-spectrum-of-a-whole-song-or-many-songs-in-batch-mode-from-start-to-end-in-one-image-bonus-remember-why-so-serious.8462/

### Multiprocessing

To speed up creation of multiple spectrograms in batch mode `soundspec` uses all available CPU cores to process the data in parallel. Reading is not parallelized but running many cores at the same time means that also the audio data for each core must be held in memory at the same time. If there are many large audio files the memory may not be sufficient. In this case run `soundspec` with option `-c <N>` to use only <N> cores or `-c 1` to force single processing mode.
