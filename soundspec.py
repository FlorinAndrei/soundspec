from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import argparse
import multiprocessing 
import os.path

# only needed for pyinstaller bug workaround
#import numpy.random.common
#import numpy.random.bounded_integers
#import numpy.random.entropy



####################################################################################

def main():
    argpar = get_argument_parser()
    args = argpar.parse_args()
    task = SoundSpec(args)
    task.process()
    
####################################################################################

def get_argument_parser():
    argpar = argparse.ArgumentParser(description="soundspec v0.1.3 - generate spectrogram from sound file")
    num_cores_avail = multiprocessing.cpu_count()
    
    # add options:
    argpar.add_argument('-b', '--batch', help='batch run, no display, save image to disk', action='store_true')
    argpar.add_argument('-c', type=int, dest='num_cores', default = num_cores_avail,
                        help='number of cores to use, default is ' + str(num_cores_avail) + 
                        ' (set to smaller value if not enough memory, or to 1 to force single core processing)')

    # add arguments:
    argpar.add_argument('audiofile', type=str, nargs='+', help='audio file (WAV) to process')

    return argpar
        
####################################################################################

class SoundSpec:
  
    def __init__(self, options):
        self.args = options
        if not self.args.batch:
            self.args.num_cores = 1 # force single core processing when not in batch mode
        self.read_file_lock = None
        self.print_lock = None
        
        # common sense limits for frequency
        self.fmin = 10
        self.fmax = 20000

        # FFT at high resolution makes way too many frequencies
        # set some lower number of frequencies we would like to keep
        # final result will be even smaller after pruning
        self.nf = 1000

    def process(self):
        num_files = len(self.args.audiofile)
        if num_files > 0 and self.args.num_cores > 1:
            num_errors = self.process_files_multi_process(self.args.audiofile)
        else:
            num_errors = self.process_files_single_process(self.args.audiofile)
        if num_errors > 0:
            self.log_message('failed for ' + str(num_errors) + ' of ' + str(num_files) + ' files.')
        else:
            if num_files > 1:
                self.log_message('succeeded for ' + str(num_files) + ' files')
                
    #-------------------------------------------------------------------------------

    def process_files_multi_process(self, audiofiles):
        process_list = []
        num_errors = 0
            
        my_read_file_lock = multiprocessing.Lock()
        my_print_lock = multiprocessing.Lock()
        
        with multiprocessing.Pool(self.args.num_cores, initializer=self.initialize_locks, initargs=(my_read_file_lock,my_print_lock,)) as pool:
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

    def initialize_locks(self, read_file_lock, print_lock):
        self.read_file_lock = read_file_lock
        self.print_lock = print_lock # multiprocessing.Lock()


    def process_file(self, audiofile):
        # read the file
        # - use a lock to prevent concurrent reads 
        if self.read_file_lock:
            self.read_file_lock.acquire()
        try:
            if not os.path.exists(audiofile):
                self.log_message(audiofile + ': not found')
                return 1
            self.log_message(audiofile + ': reading ...')
            sf, audio = wavfile.read(audiofile)
        except:
            self.log_message(audiofile + ': error reading audio data')
            return 1
        finally:
            if self.read_file_lock:
                self.read_file_lock.release()
        # starting from here the code runs concurrently in batch mode
        
        if sf < self.fmin:
          self.log_message(audiofile + ': Sampling frequency too low')
          return 1

        self.log_message(audiofile + ': processing ...')

        # convert to mono
        sig = np.mean(audio, axis=1)

        # vertical resolution (frequency)
        # number of points per segment; more points = better frequency resolution
        # if equal to sf, then frequency resolution is 1 Hz
        npts = int(sf)

        # horizontal resolution (time)
        # fudge factor to keep the number of frequency samples close to 1000
        # (assuming an image width of about 1000 px)
        # negative values ought to be fine
        # this needs to change if image size becomes parametrized
        winfudge = 1 - ((np.shape(sig)[0] / sf) / 1000) # TODO: should we replace 1000 by self.nf?

        self.log_message(audiofile + ': calculating FFT ...')
        f, t, Sxx = signal.spectrogram(sig, sf, nperseg=npts, noverlap=int(winfudge * npts))

        # generate an exponential distribution of frequencies
        # (as opposed to the linear distribution from FFT)
        b = self.fmin - 1
        a = np.log10(self.fmax - self.fmin + 1) / (self.nf - 1)
        freqs = np.empty(self.nf, int)
        for i in range(self.nf):
          freqs[i] = np.power(10, a * i) + b
        # list of frequencies, exponentially distributed:
        freqs = np.unique(freqs)

        # delete frequencies lower than fmin
        fnew = f[f >= self.fmin]
        cropsize = f.size - fnew.size
        f = fnew
        Sxx = np.delete(Sxx, np.s_[0:cropsize], axis=0)

        # delete frequencies higher than fmax
        fnew = f[f <= self.fmax]
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

        self.log_message(audiofile + ': generating the image ...')
        plt.pcolormesh(t, f, np.log10(Sxx))
        plt.ylabel('f [Hz]')
        plt.xlabel('t [sec]')
        plt.yscale('symlog')
        plt.ylim(self.fmin, self.fmax)

        # TODO: make this depend on fmin / fmax
        # right now I'm assuming a range close to 10 - 20000
        yt = np.arange(10, 100, 10)
        yt = np.concatenate((yt, 10 * yt, 100 * yt, 1000 * yt))
        yt = yt[yt <= self.fmax]
        yt = yt.tolist()
        plt.yticks(yt)

        plt.grid(True)
        if self.args.batch:
            image_file = audiofile + '.png'
            self.log_message(audiofile + ': create spectrogram ' + image_file)
            plt.savefig(image_file, dpi=200)
        else:
            plt.show()
        return 0

    #-------------------------------------------------------------------------------

    def log_message(self, msg):
        if self.print_lock:
            self.print_lock.acquire()
        try:
            print(msg)
        finally:
            if self.print_lock:
                self.print_lock.release()

          
####################################################################################

if __name__ == "__main__":
    main()

