from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
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
    argpar = argparse.ArgumentParser(description="soundspec v0.1.2 - generate spectrogram from sound file")

    # add options:
    argpar.add_argument('-b', '--batch', help='batch run, no display, save image to disk', action='store_true')
    argpar.add_argument('-s', '--singleproc', action='store_true', dest='forceSingleProcessing', default=False,
                        help='force single processing in batch mode on system with multiple cores (use if not enough memory)')

    # add arguments:
    argpar.add_argument('audiofile', type=str, nargs='+', help='audio file (WAV) to process')

    return argpar
        
####################################################################################

class SoundSpec:
  
    def __init__(self, options):
        self.args = options
        
        # common sense limits for frequency
        self.fmin = 10
        self.fmax = 20000

        # FFT at high resolution makes way too many frequencies
        # set some lower number of frequencies we would like to keep
        # final result will be even smaller after pruning
        self.nf = 1000

    def process(self):
        num_files = len(self.args.audiofile)
        if num_files > 0 and not self.args.forceSingleProcessing and self.args.batch == True:
            num_errors = self.process_files_multi_process(self.args.audiofile)
        else:
            num_errors = self.process_files_single_process(self.args.audiofile)
        if num_errors > 0:
            print('failed for ' + str(num_errors) + ' of ' + str(num_files) + ' files.')
        else:
            if num_files > 1:
                print('succeeded for ' + str(num_files) + ' files')
                
    def process_files_multi_process(self, audiofiles):
        num_cpus = multiprocessing.cpu_count()
        process_list = []
        num_errors = 0
            
        with multiprocessing.Pool(num_cpus) as pool:
            for audiofile in audiofiles:
                if os.path.exists(audiofile):
                    process_data = self.read_file(audiofile)
                    process = pool.apply_async(self.process_file, (process_data,))
                    process_list.append(process)
                else:
                    print(audiofile + ': not found')
                    num_errors += 1

            for process in process_list:
                num_errors += process.get()

        return num_errors

                
    def process_files_single_process(self, audiofiles):
        num_errors = 0
        for audiofile in audiofiles:
            if os.path.exists(audiofile):
                process_data = self.read_file(audiofile)
                num_errors += self.process_file(process_data)
            else:
                print(audiofile + ': not found')
                num_errors += 1
        return num_errors


    #-------------------------------------------------------------------------------

    def read_file(self, audiofile):
        print(audiofile + ': reading ...')
        process_data = ProcessData(audiofile)
        process_data.sf, process_data.audio = wavfile.read(audiofile)
        return process_data
      

    def process_file(self, process_data):
        print(process_data.audiofile + ': processing ...')

        if process_data.sf < self.fmin:
          print(process_data.audiofile + ': Sampling frequency too low.')
          return 1

        # convert to mono
        sig = np.mean(process_data.audio, axis=1)

        # vertical resolution (frequency)
        # number of points per segment; more points = better frequency resolution
        # if equal to sf, then frequency resolution is 1 Hz
        npts = int(process_data.sf)

        # horizontal resolution (time)
        # fudge factor to keep the number of frequency samples close to 1000
        # (assuming an image width of about 1000 px)
        # negative values ought to be fine
        # this needs to change if image size becomes parametrized
        winfudge = 1 - ((np.shape(sig)[0] / process_data.sf) / 1000)

        print(process_data.audiofile + ': calculating FFT ...')
        f, t, Sxx = signal.spectrogram(sig, process_data.sf, nperseg=npts, noverlap=int(winfudge * npts))

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

        print(process_data.audiofile + ': generating the image ...')
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
            image_file = process_data.audiofile + '.png'
            print(process_data.audiofile + ': create spectrogram ' + image_file)
            plt.savefig(image_file, dpi=200)
        else:
            plt.show()
        return 0


class ProcessData:
        def __init__(self, filename):
          self.audiofile = filename

####################################################################################

if __name__ == "__main__":
    main()

