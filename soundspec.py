#!/usr/bin/env python3

from scipy.io import wavfile
from scipy import signal
from scipy.stats import mstats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import argparse
import multiprocessing 
import os.path
import os
import subprocess
import time
import wave as wv
import math

# only needed for pyinstaller bug workaround
#import numpy.random.common
#import numpy.random.bounded_integers
#import numpy.random.entropy

####################################################################################

class SoundSpecLocks:

    def __init__(self):
        self.print_lock = None
        self.read_file_lock = None
        self.plot_file_lock = None
        
            
    def enable(self):
        self.print_lock = multiprocessing.Lock()
        self.read_file_lock = multiprocessing.Lock()
        self.plot_file_lock = multiprocessing.Lock()    # probably not needed 


locks = SoundSpecLocks()

####################################################################################

def main():
    argpar = get_argument_parser()
    args = argpar.parse_args()
    task = SoundSpec(args)
    task.process()
    
####################################################################################

def get_argument_parser():
    argpar = argparse.ArgumentParser(description="soundspec v0.1.7 - generate spectrogram from sound file")
    num_cores_avail = multiprocessing.cpu_count()
    
    # add options:
    argpar.add_argument('-b', '--batch', help='batch run, no display, save image to disk', action='store_true')
    argpar.add_argument('-c', type=int, dest='num_cores', default = num_cores_avail,
                        help='number of cores to use, default is ' + str(num_cores_avail) + 
                        ' (set to smaller value if not enough memory, or to 1 to force single core processing)')
    argpar.add_argument('-d', '--debug', type=int, dest='debug_level', help='set debug level', default=0)
    argpar.add_argument('-l', '--linear', dest='use_linear_amplitude', help='use linear amplitude', action='store_true')
    argpar.add_argument('-m', '--maximum', dest='use_maximum', help='use maximum instead of skipping frequencies', action='store_true')
    argpar.add_argument('-p', '--ppi', type=int, help='set print resolution for batch mode (default is 200)', default=200)
    argpar.add_argument('-r', '--resolution', type=int, help='set resolution (default is 1000)', default=1000)
    argpar.add_argument('-w', '--window', help='set window for FFT (default is blackmanharris)', default='blackmanharris')

    # add arguments:
    argpar.add_argument('audiofile', type=str, nargs='+', 
                        help='audio file (WAV) to process, or a directory containing audio files')

    return argpar
        
####################################################################################


class SoundSpec:
  
    def __init__(self, options):
        self.args = options
        self.logger = SoundSpecLogger(options)
        if not self.args.batch:
            self.args.num_cores = 1 # force single core processing when not in batch mode
        
        # FFT at high resolution makes way too many frequencies
        # set some lower number of frequencies we would like to keep
        # final result will be even smaller after pruning
        self.nf = self.args.resolution
        self.known_extensions = ['.wav']  # WAV files

        # settings for ffmpeg
        self.ffmpeg = 'ffmpeg'  # set full path to ffmpeg if required 
        self.ffmpeg_loglevel = 'error'
        if self.ffmpeg != None:
            # add all file types handled by ffmpeg: (incomplete)
            self.known_extensions.append('.flac')
            self.known_extensions.append('.mp3')
            self.known_extensions.append('.mp4')
            self.known_extensions.append('.m4a')
            self.known_extensions.append('.mkv')
            self.known_extensions.append('.avi')
            self.known_extensions.append('.ogg')
            self.known_extensions.append('.webm')


    def process(self):
        audio_files = self.get_list_of_files(self.args.audiofile)
        num_files = len(audio_files)
        if num_files > 0 and self.args.num_cores > 1:
            num_errors = self.process_files_multi_process(audio_files)
        else:
            num_errors = self.process_files_single_process(audio_files)
        if num_errors > 0:
            self.logger.log_message('failed for ' + str(num_errors) + ' of ' + str(num_files) + ' files.')
        else:
            if num_files > 1:
                self.logger.log_message('succeeded for ' + str(num_files) + ' files')
                
    #-------------------------------------------------------------------------------

    def get_list_of_files(self, audiofiles):
        list_of_files = []
        for file in audiofiles:
            if os.path.isdir(file):
                self.add_files_in_dir(file, list_of_files)
            else:
                if self.is_audio_file(file):
                    list_of_files.append(file)
                else:
                    self.logger.log_message('ignored file ' + file + ': file type not supported or no audio file')
        return list_of_files
    
    
    def process_files_multi_process(self, audiofiles):
        process_list = []
        num_errors = 0
        locks.enable()
        
        with multiprocessing.Pool(self.args.num_cores, ) as pool:
            for audiofile in audiofiles:
                process = pool.apply_async(self.process_file, (audiofile,))
                process_list.append(process)

            for process in process_list:
                num_errors += process.get()

        return num_errors

                
    def process_files_single_process(self, audiofiles):
        num_errors = 0
        for audiofile in audiofiles:
            num_errors += self.process_file(audiofile)
        return num_errors

    #-------------------------------------------------------------------------------

    def add_files_in_dir(self, directory, list_of_files):
        for root, dirs, files in os.walk(directory, topdown = True):
            for name in files:
                if self.is_audio_file(name):
                    path = os.path.join(root, name)
                    list_of_files.append(path)
        

    def is_audio_file(self, filename):
        root, ext = os.path.splitext(filename)
        for known_ext in self.known_extensions:
            if ext.lower() == known_ext:
                return True
        return False
    
    
    def process_file(self, audiofile):
        audiofile_reader = AudioFileReader(self.logger)
        sf, bw, audio = audiofile_reader.read(audiofile)
        # starting from here the code runs concurrently in batch mode
        if sf != None: 
            spectrogram_creator = SpectrogramCreator(self.args, self.logger)
            spectrogram_creator.create(audiofile, sf, bw, audio)
            return 0
        else:
            return 1


class AudioFileReader:
    
    def __init__(self, logger):
        self.logger = logger
        self.known_extensions = ['.wav']  # WAV files
        
        # settings for ffmpeg
        self.ffmpeg = 'ffmpeg'  # set full path to ffmpeg if required 
        self.ffmpeg_loglevel = 'error'
        if self.ffmpeg != None:
            # add all file types handled by ffmpeg: (incomplete)
            self.known_extensions.append('.flac')
            self.known_extensions.append('.mp3')
            self.known_extensions.append('.mp4')
            self.known_extensions.append('.m4a')
            self.known_extensions.append('.mkv')
            self.known_extensions.append('.avi')
            self.known_extensions.append('.ogg')
            self.known_extensions.append('.webm')


    def read(self, audiofile):
        # read the file
        # - use a lock to prevent concurrent reads (if available)
        if locks.read_file_lock:
            locks.read_file_lock.acquire()
        wav_file = None
        
        try:
            if not os.path.exists(audiofile):
                self.logger.log_message(audiofile + ': not found')
                return None, None, None
            if self.is_wav_file(audiofile):
                wav_file = audiofile
            else:    
                # convert audiofile into wav_file:
                wav_file = self.convert_to_wav(audiofile)
                if wav_file == None:
                    return None, None, None
                if not os.path.exists(wav_file):
                    self.logger.log_message(audiofile + ': failed to convert into wav format')
                    return None, None, None
                    
            self.logger.log_message(audiofile + ': reading WAV file ...')
            sf, audio = wavfile.read(wav_file) # sf = samplerate in Hz, audio.shape[] = #samples, [#channels (if > 1)]
            
            num_samples = audio.shape[0]
            num_channels = 1;
            if len(audio.shape) > 1:
                num_channels = audio.shape[1]
            run_time = num_samples / sf
            run_time_str = time.strftime("%H:%M:%S", time.gmtime(run_time))
            bit_width = self.get_bit_width(wav_file)
            self.logger.log_message(audiofile + ': ' + str(num_channels) + ' channel(s) at ' + str(sf/1000) + ' kHz samplerate with ' + run_time_str + ' runtime')
        
        except:
            self.logger.log_message(audiofile + ': error reading audio data')
            return None, None, None
        finally:
            if wav_file != None and wav_file != audiofile:
                os.unlink(wav_file)
            if locks.read_file_lock:
                locks.read_file_lock.release()
        return sf, bit_width, audio

    #-------------------------------------------------------------------------------

    def get_bit_width(self, audiofile):
        try:
            wavefile = wv.open(audiofile, 'rb')
            bw = 8 * wavefile.getsampwidth()
        except:
            bw = 16
        finally:
            wavefile.close()
        return bw
    
        
    def is_wav_file(self, filename):
        root, ext = os.path.splitext(filename)
        if ext.lower() == '.wav':
            return True
        return False


    def convert_to_wav(self, audiofile):
        # convert audiofile into wav_file:
        self.logger.log_message(audiofile + ': converting into wav format ...')
        wav_file = audiofile + '_tmp-soundspec.wav'
        if os.path.exists(wav_file):
            os.unlink(wav_file)
        
        try:
            args = [self.ffmpeg, '-loglevel', self.ffmpeg_loglevel, '-nostdin', '-i', audiofile, wav_file]
            completed_process = subprocess.run(args, capture_output=True, check=True)
        except subprocess.SubprocessError as ex:
            self.logger.log_message(audiofile + ': failed to convert into wav format: ' + ex.stderr.decode())
            return None
        except:
            self.logger.log_message(audiofile + ': failed to convert into wav format: unknown exeption')
            return None
        return wav_file
    

class SpectrogramCreator:
  
    def __init__(self, options, logger):
        self.args = options
        self.logger = logger
        
        # FFT at high resolution makes way too many frequencies
        # set some lower number of frequencies we would like to keep
        # final result will be even smaller after pruning
        self.nf = self.args.resolution

    
    def create(self, audiofile, sf, bw, audio):
        # common sense limits for frequency
        fmin = 10
        if sf < fmin:
            self.logger.log_message(audiofile + ': Sampling frequency too low')
            return 1

        self.logger.log_message(audiofile + ': processing ...')
        fmax = self.determine_max_freq_to_show(sf)

        # convert to mono
        sig = np.mean(audio,axis = 1) if (audio.ndim>=2) else audio 
        self.logger.debug_message(2, 'audio range: [' + str(np.amin(np.amin(sig))) + ' .. ' + str(np.amax(np.amax(sig))) + ']')
        
        # vertical resolution (frequency)
        # number of points per segment; more points = better frequency resolution
        # if equal to sf, then frequency resolution is 1 Hz
        npts = int(sf)

        # horizontal resolution (time)
        # fudge factor to keep the number of frequency samples close to self.nf
        # (assuming an image width of about self.nf px)
        # negative values ought to be fine
        num_samples = audio.shape[0]
        run_time = num_samples / sf
        winfudge = 1 - (run_time / self.nf)
        num_overlap = int(winfudge * npts)

        window_correction = 2.831593 # for blackman harris window
        # create the spectrogram, returns:
        # - f = [0..sf/2], 
        # - t = [0.5 .. run_time-0.5],
        # - Sxx = array[f.size, t.size]
        self.logger.log_message(audiofile + ': calculating FFT ...')
        f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts, noverlap=num_overlap, window = self.args.window, mode='magnitude')
        self.logger.debug_message(6, 'f.shape: ' + str(f.shape))
        self.logger.debug_message(6, 't.shape: ' + str(t.shape))
        self.logger.debug_message(6, 'Sxx.shape: ' + str(Sxx.shape))
        self.logger.debug_message(2, 'spectrum range: [' + str(np.amin(np.amin(Sxx))) + ' .. ' + str(np.amax(np.amax(Sxx))) + ']')

        self.logger.log_message(audiofile + ': downscaling spectra ...')
        # generate an exponential distribution of frequencies
        # (as opposed to the linear distribution from FFT)
        freqs = self.exponential_distribution_of_frequencies(fmin, fmax)

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
        self.logger.debug_message(1, 'Sxx.shape: ' + str(Sxx.shape))

        # find FFT frequencies closest to calculated exponential frequency distribution
        findex = []
        for i in range(freqs.size):
            f_ind = (np.abs(f - freqs[i])).argmin()
            findex.append(f_ind)
        self.logger.debug_message(2, 'findex: len=' + str(len(findex)) + ': ' + str(findex))

        # keep only frequencies closest to exponential distribution
        # this is usually a massive cropping of the initial FFT data
        fnew = []
        for i in findex:
            fnew.append(f[i])
        f = np.asarray(fnew)

        if self.args.use_maximum:
            num_errors, Sxxnew = self.scale_down_with_maxima(freqs.size, findex, Sxx)
            if num_errors > 0:
                return 1
            Sxx = Sxxnew * window_correction
        else:
            # strip unused frequencies from Sxx
            Sxxnew = Sxx[findex, :]
            Sxx = Sxxnew * window_correction

        self.logger.debug_message(1, Sxx.shape)
        if not self.args.use_linear_amplitude:
            Sxx = self.convert_into_db(Sxx, npts, bw)
            self.logger.debug_message(2, 'spectrum range in dB: [' + str(np.amax(np.amax(Sxx))) + ' .. ' + str(np.amin(np.amin(Sxx))) + ']')

        self.logger.debug_message(3, 'Sxx:\n' + str(Sxx))

        self.logger.log_message(audiofile + ': creating projection ...')
        projection = np.amax(Sxx, axis=1)

        self.logger.log_message(audiofile + ': generating the image ...')
        plotter = SpectrogramPlotter(self.args, fmin, fmax, self.logger)
        plotter.plot_spectrogram(t, f, Sxx, projection, audiofile)
        return 0
    
    #-------------------------------------------------------------------------------

    def determine_max_freq_to_show(self, sf):
        fmax = 20000
        if sf >= 48000:
            fmax = 22000
        if sf >= 88000:
            fmax = 40000
        if sf >= 176000:
            fmax = 80000
        if sf >= 352000:
            fmax = 160000
        if sf >= 704000:
            fmax = 320000
        return fmax


    def exponential_distribution_of_frequencies(self, fmin, fmax):
        # generate an exponential distribution of frequencies
        # (as opposed to the linear distribution from FFT)
        b = fmin - 1
        a = np.log10(fmax - fmin + 1) / (self.nf - 1)
        freqs = np.empty(self.nf, int)
        for i in range(self.nf):
            freqs[i] = np.power(10, a * i) + b
        # list of frequencies, exponentially distributed:
        freqs = np.unique(freqs)    # remove duplicates
        self.logger.debug_message(1, 'freqs.size: ' + str(freqs.size))
        return freqs
    
        
    def scale_down_with_maxima(self, num_freqs, findex, Sxx):
        # determine range for each frequency bin findex[i]
        findex_begin = []
        findex_end = []
        for i in range(num_freqs):
            if i == 0:
                begin = 0                                       # at 1st point
                findex_begin.append(begin)

                end = self.get_mean(findex[0 : 2])              # geometric mean of 1st two points
                end = min(int(end + 0.5), findex[1] - 1)        # limit to valid range (no overlap with next range)
                end = max(end, 0)
                findex_end.append(end)
            else:
                if i < (num_freqs - 1):
                    begin = findex_end[i-1] + 1                 # at 1st point after previous range
                    findex_begin.append(begin)

                    end = self.get_mean(findex[i : i+2])        # geometric mean of current and next point
                    end = min(int(end+0.5), findex[i+1] - 1)    # limit to valid range (no overlap with next range
                    end = max(end, findex[i])
                    findex_end.append(end)
                else:
                    begin = findex_end[i-1] + 1                 # at 1st point after previous range
                    findex_begin.append(begin)

                    end = findex[i]                             # at last point
                    findex_end.append(end)
        self.logger.debug_message(6, findex_begin)
        self.logger.debug_message(6, findex_end)

        # sanity check
        num_errors = self.downscale_sanity_check(num_freqs, findex, findex_begin, findex_end)
        if num_errors > 0:
            return num_errors, None

        self.logger.debug_message(2, 'Sxx.shape: ' + str(Sxx.shape))
        maximum_spectra = np.empty((num_freqs, Sxx.shape[1]))  # 85 frequencies in 91 spectra
        self.logger.debug_message(2, 'maximum_spectra.shape :' + str(maximum_spectra.shape))
        for i in range(num_freqs):
            # select those spectra which shall be used to find the maximum for each frequency
            ssx_range = range(findex_begin[i], findex_end[i] + 1)
            self.logger.debug_message(4, 'ssx_range             : ' + str(ssx_range))
            selected_spectra = Sxx[ssx_range, :]
            self.logger.debug_message(4, 'selected_spectra.shape: ' + str(selected_spectra.shape))
            self.logger.debug_message(5, 'selected_spectra: ' + str(selected_spectra))
            maximum_spectra[i] = np.amax(selected_spectra, axis=0)

        return 0, maximum_spectra
    
    #-------------------------------------------------------------------------------


    def convert_into_db(self, Sxx, npts, bit_width):
        max_value = 2 ** (bit_width - 1)
        Sxx_dB = 20 * np.log10(Sxx / max_value)

        # set lower limit according to FFT gain and bit width
        fft_gain = self.get_fft_gain(npts)
        min_value_dB = - (bit_width * 6 + fft_gain)
        Sxx_dB[Sxx_dB < min_value_dB] = min_value_dB
        return Sxx_dB   
        
        
    #-------------------------------------------------------------------------------

    def downscale_sanity_check(self, num_freqs, findex, findex_begin, findex_end):
        for i in range(num_freqs):
            if findex_begin[i] > findex[i]:
                self.logger.log_message(audiofile + ': findex[' + str(i) + ']: begin error')
                return 1
            if findex_end[i] < findex[i]:
                self.logger.log_message(audiofile + ': findex[' + str(i) + ']: end error')
                return 1
            if i == 0:
                if findex_end[i] >= findex_begin[i+1]:
                    self.logger.log_message(audiofile + ': findex[' + str(i) + ']: end overlap error')
                    return 1
            else:
                if i < (num_freqs - 1):
                    if findex_begin[i] <= findex_end[i-1]:
                        self.logger.log_message(audiofile + ': findex[' + str(i) + ']: begin overlap error')
                        return 1
                    if findex_end[i] >= findex_begin[i+1]:
                        self.logger.log_message(audiofile + ': findex[' + str(i) + ']: end overlap error')
                        return 1
                else:
                    if findex_begin[i] <= findex_end[i-1]:
                        self.logger.log_message(audiofile + ': findex[' + str(i) + ']: begin overlap error')
                        return 1
        return 0

    def get_fft_gain(self, npts):
        fft_gain = 3 * math.log(npts) / math.log(2) - 3
        return fft_gain
 

    def get_mean(self, values):
        try:
            np.seterr(divide = 'raise') 
            mean_value = mstats.gmean(values) # may fail with "RuntimeWarning: divide by zero encountered in log"
        except:
            mean_value = np.mean(values)  # use arithmetic mean if geometric mean fails, e.g. for [0, 1]
        finally:
            np.seterr(divide = 'warn')
        return mean_value
       

class SpectrogramPlotter:
    
    def __init__(self, options, fmin, fmax, logger):
        self.args = options
        self.fmin = fmin
        self.fmax = fmax
        self.logger = logger


    def plot_spectrogram(self, t, f, Sxx, projection, audiofile):
        if locks.plot_file_lock:
            locks.plot_file_lock.acquire()
        try:
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 2,  width_ratios = [2, 6], wspace = 0.02)

            self.__plot_projection(fig, gs, f, projection)
            self.__plot_spectrogram(fig, gs, t, f, Sxx)

            plt.suptitle(os.path.basename(audiofile))
            if self.args.batch:
                image_file = audiofile + '.png'
                self.logger.log_message(audiofile + ': create spectrogram ' + os.path.basename(image_file))
                plt.savefig(image_file, dpi=self.args.ppi)
            else:
                plt.show()
        finally:
            if locks.plot_file_lock:
                locks.plot_file_lock.release()

    #-------------------------------------------------------------------------------

    def __plot_projection(self, fig, gs, f, projection):
        ax_proj = fig.add_subplot(gs[0])
            
        # rotate plot by 90 degree
        # see https://stackoverflow.com/questions/22540449/how-can-i-rotate-a-matplotlib-plot-through-90-degrees
        proj_base = ax_proj.transData
        rotation = transforms.Affine2D().rotate_deg(90)
        ax_proj.plot(f, projection, transform = rotation + proj_base)
            
        ax_proj.set_title('Projection')
        self.__set_projection_xaxis(ax_proj)
        self.__set_projection_yaxis(ax_proj)
        ax_proj.grid(True)


    def __plot_spectrogram(self, fig, gs, t, f, Sxx):
        ax_spgr = fig.add_subplot(gs[1])
        ax_spgr.pcolormesh(t, f, Sxx)
        self.__set_spectrogram_xaxis(ax_spgr, t)
        self.__set_spectrogram_yaxis(ax_spgr)
        ax_spgr.grid(True)
        ax_spgr.set_title('Spectrogram')
            
    #-------------------------------------------------------------------------------
        
    def __set_projection_xaxis(self, ax):
        if self.args.use_linear_amplitude:
            ax.set_xticks([])
        else:
            ax.set_xticks(     [0,   20,   40,  60,   80, 100,   120])
            ax.set_xticklabels(['0', '', '-40', '', '-80', '', '-120'])
        ax.set_xlabel('Peak Amplitude')
    
    
    def __set_projection_yaxis(self, ax):
        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylim(self.fmin, self.fmax)
        ax.set_yscale('symlog')
        # ax.get_yaxis().tick_right()
        ticks = self.get_frequency_ticks()
        labels = self.get_frequency_labels(ticks, empty=False)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
    
    
    def __set_spectrogram_xaxis(self, ax, t):
        time_s = t[t.size-1]
        ax.set_xlabel(self.get_time_label(time_s))

        ticks = self.get_time_ticks(time_s)
        if ticks != None:
            labels = self.get_time_labels(time_s, ticks)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            

    def __set_spectrogram_yaxis(self, ax):
#        ax.set_ylabel('Frequency [Hz]')
        ax.set_yscale('symlog')
        ax.set_ylim(self.fmin, self.fmax)
        
        ticks = self.get_frequency_ticks()
        labels = self.get_frequency_labels(ticks, empty=True)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

    #-------------------------------------------------------------------------------

    def get_time_label(self, time_s):
        if time_s <= 60:
            return 'Time [sec]'
        if time_s <= (5 * 60):
            return 'Time [min:sec]'
        return 'Time [min]'


    def get_time_ticks(self, time_s):
        if time_s <= 60:
            return None # use standard x-axis
        num_minutes = time_s / 60
        step = 10               # 0:10 ...
        if num_minutes > 1.7:
            step = 20           # 0:20 ...
        if num_minutes > 3:
            step = 30           # 0:30 ...
        if num_minutes > 5:
            step = 60           # 1:00 ...
        if num_minutes > 10:
            step = 120          # 2:00 ...
        if num_minutes > 30:
            step = 300          # 5:00
        if num_minutes > 60:
            step = 600          # 10:00
        if num_minutes > 120:
            step = 1200         # 20:00
        ticks = np.arange(step, time_s, step)
        return ticks.tolist()
    
    
    def get_time_labels(self, time_s, ticks):
        num_minutes = time_s / 60
        if num_minutes > 5: 
            time_format= "%M"       # returns mm
        else:
            time_format = "%M:%S"   # returns mm:ss
        labels = []
        for tick in ticks:
            label = time.strftime(time_format, time.gmtime(tick))
            if label[0] == '0':
                label = label[1:]   # strip leading 0
            labels.append(label)
        return labels
    
    
    def get_frequency_ticks(self):
        yt = np.arange(10, 100, 10)                                         # creates [10, 20, 30, ... 80, 90]
        yt = np.concatenate((yt, 10 * yt, 100 * yt, 1000 * yt, 10000 * yt)) # extend to 100, 1k, 10k, 100k
        yt = yt[yt <= self.fmax]
        ticks = yt.tolist()
        return ticks


    def get_frequency_labels(self, ticks, empty):
        labels = []
        for tick in ticks:
            if empty == True:
                label = ''
            else:
                if tick >= 1000:
                    label = str(int(tick/1000)) + 'k'
                else:
                    label = str(int(tick))
                if label[0] != '1' and label[0] != '2' and label[0] != '5':
                    label = ''
            labels.append(label)
        return labels


class SoundSpecLogger:
  
    def __init__(self, options):
        self.args = options
        self.print_lock = None
        
        
    def initialize_lock(self, print_lock):
        self.print_lock = print_lock


    def log_message(self, msg):
        if locks.print_lock:
            locks.print_lock.acquire()
        try:
            print(msg)
        finally:
            if locks.print_lock:
                locks.print_lock.release()


    def debug_message(self, level, msg):
        if self.args.debug_level > level:
            self.log_message(msg)
          


####################################################################################

if __name__ == "__main__":
    main()

