import os.path
import numpy as np
import argparse
from glob import glob
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from sklearn.mixture import GMM
from audiotools import *
from argparse import ArgumentParser
#from scikits.learn.gmm import GMM
from MFCC import melScaling

#######################################################################
# some settings

framelen = 1024
#fs = 44100.0
#fs=16000.0
fs=0
verbose = True

#######################################################################
# main class

class gmm1:
    """
    USAGE EXAMPLE:
    In this hypothetical example we train on four audio files, labelled as either 'usa' or 'uk', and then test on a separate audio file:

    from gmmfl import gmm1
            ll = gmm.eval(features)[0]
    model = gmm1("wavs/training", {'karen01.wav':'usa', 'john01.wav':'uk', 'steve02.wav':'usa', 'joe03.wav':'uk'})
    model.classify('wavs/testing/hubert01.wav')
    """
    def get_info(self,audio_file):
        """create a dictionary of information for the audiofile object."""
        info = {}
        info["channels"] = audio_file.channels()
        info["channel_mask"] = audio_file.channel_mask()
        info["bits"] = audio_file.bits_per_sample()
        info["sample_rate"] = audio_file.sample_rate()
        info["frames"] = audio_file.total_frames()
        info["length"] = audio_file.seconds_length()
        info["seekable"] = audio_file.seekable()
        info["verified"] = audio_file.verify()
        info["chunks"] = audio_file.has_foreign_wave_chunks()
        info["available"] = audio_file.available(BIN)
        info["header"], info["footer"] = audio_file.wave_header_footer()

        global fs
        fs=info["sample_rate"]

        print "No. of Channels:\t\t", info["channels"]
        print "Channel mask:\t\t\t", info["channel_mask"]
        print "Bits per sample:\t\t", info["bits"], "BIT"
        print "Sample Rate:\t\t\t", (info["sample_rate"]/1000.0), "k"
        print "Number of Frames:\t\t", info["frames"]
        print "Audio Length:\t\t\t", info["length"], "seconds"
        print "Audio File Seekable?:\t\t", info["seekable"]
        print "File has foreign chunks?:\t", info["chunks"]
        print "Correct Binaries present?:\t", info["available"]

        return info

    def __init__(self, wavfolder, trainingdata):
        """Initialise the classifier and train it on some WAV files.
        'wavfolder' is the base folder, to be prepended to all WAV paths.
        'trainingdata' is a dictionary of wavpath:label pairs."""

        #open audio file as an AudioFile object
        for wavpath in trainingdata:
            audio_file = open((os.path.join(wavfolder, wavpath)))
            file_info=self.get_info(audio_file)

            #Creates a WaveReader object from the AudioFile Object
            pcm_data = audio_file.to_pcm()

            #Creates a FrameList object from WaveReader object. Currently reads all
            #frames in file
            frame_list = pcm_data.read(file_info["frames"])

            #Convert samples to floats (-1.0 - +1.0)
            float_frame_list = frame_list.to_float()

            #eventually do some signal processing here...

            #Convert back to integer FrameList
            output_framelist = float_frame_list.to_int(file_info["bits"])

            #now back to raw bytes
            output_data = output_framelist.to_bytes(False, True)

        self.mfccMaker = melScaling(int(fs), framelen/2, 40)
        self.mfccMaker.update()

        allfeatures = {wavpath:self.file_to_features(os.path.join(wavfolder, wavpath)) for wavpath in trainingdata}

        ###print('ALLfeatures:',allfeatures)####checking features###
        # Determine the normalisation stats, and remember them
        allconcat = np.vstack(list(allfeatures.values()))#stack arrays in sequence vertically(row wise))
        self.means = np.mean(allconcat, 0)#compute mean
        self.invstds = np.std(allconcat, 0)#compute the standard deviation
        for i,val in enumerate(self.invstds):
            if val == 0.0:
                self.invstds[i] = 1.0
            else:
                self.invstds[i] = 1.0 / val

        # For each label, compile a normalised concatenated list of features
        aggfeatures = {}
        for wavpath, features in allfeatures.items():
            label = trainingdata[wavpath]
            normed = self.__normalise(features)
            if label not in aggfeatures:
                aggfeatures[label] = normed
            else:
                aggfeatures[label] = np.vstack((aggfeatures[label], normed))

        # For each label's aggregated features, train a GMM and remember it
        self.gmms = {}
        for label, aggf in aggfeatures.items():
            if verbose: print("    Training a GMM for label %s, using data of shape %s" % (label, str(np.shape(aggf))))
            self.gmms[label] = GMM(n_components=10) # , cvtype='full')
            self.gmms[label].fit(aggf)
        if verbose: print("  Trained %i classes from %i input files" % (len(self.gmms), len(trainingdata)))


    def __normalise(self, data):
        "Normalises data using the mean and stdev of the training data - so that everything is on a common scale."
        return (data - self.means) * self.invstds

    def classify(self, wavpath):
        "Specify the path to an audio file, and this returns the max-likelihood class, as a string label."
        features = self.__normalise(self.file_to_features(wavpath))
        # For each label GMM, find the overall log-likelihood and choose the strongest
        bestlabel = ''
        bestll = -9e99
        for label, gmm in self.gmms.items():
#            ll = gmm.eval(features)[0]
            ll = gmm.score_samples(features)[0]#used instead of eval to compute likelihood per sample
            ll = np.sum(ll)
            if ll > bestll:
                bestll = ll
                bestlabel = label
        return bestlabel

    def file_to_features(self, wavpath):
        "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
        if verbose: print("Reading %s" % wavpath)
        if not os.path.isfile(wavpath): raise ValueError("path %s not found" % wavpath)
        sf = Sndfile(wavpath, "r")
        #if (sf.channels != 1) and verbose: print(" Sound file has multiple channels (%i) - channels will be mixed to mono." % sf.channels)
        #if sf.samplerate != fs:         raise ValueError("wanted sample rate %g - got %g." % (fs, sf.samplerate))
        window = np.hamming(framelen)#check here
        features = []
        while(True):
            try:
                chunk = sf.read_frames(framelen, dtype=np.float32)
                if len(chunk) != framelen:
                    print("Not read sufficient samples - returning")
                    break
                if sf.channels != 1:
                    chunk = np.mean(chunk, 1) # mixdown
                framespectrum = np.fft.fft(window * chunk)
                magspec = abs(framespectrum[:framelen/2])

                # do the frequency warping and MFCC computation
                melSpectrum = self.mfccMaker.warpSpectrum(magspec)
                melCepstrum = self.mfccMaker.getMFCCs(melSpectrum,cn=True)
                melCepstrum = melCepstrum[1:]   # exclude zeroth coefficient
                melCepstrum = melCepstrum[:13] # limit to lower MFCCs

                framefeatures = melCepstrum   # todo: include deltas? that can be your homework.

                features.append(framefeatures)
            except RuntimeError:
                break
        sf.close()
        return np.array(features)

#######################################################################
def trainAndTest(trainpath, trainwavs, testpath, testwavs):
    "Handy function for evaluating your code: trains a model, tests it on wavs of known class. Returns (numcorrect, numtotal, numclasses)."
    print("TRAINING")
    model = gmm1(trainpath, trainwavs)
    print("TESTING")
    ncorrect = 0
    for wavpath,label in testwavs.items():
        result = model.classify(os.path.join(testpath, wavpath))
        if verbose: print(" inferred: %s" % result)
        if result == label:
            ncorrect += 1
    return (ncorrect, len(testwavs), len(model.gmms))

#######################################################################
# If this file is invoked as a script, it carries out a simple runthrough
# of training on some wavs, then testing, with classnames being the start of the filenames
if __name__ == '__main__':

    # Handle the command-line arguments for where the train/test data comes from:
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainpath', default='wavs', help="Path to the WAV files used for training")
    parser.add_argument('-T', '--testpath',                  help="Path to the WAV files used for testing")
    parser.add_argument('-q', dest='quiet', action='store_true', help="Be less verbose, don't output much text during processing")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-c', '--charsplit',  default='_',    help="Character used to split filenames: anything BEFORE this character is the class")
    group.add_argument('-n', '--numchars' ,  default=0  ,    help="Instead of splitting using 'charsplit', use this fixed number of characters from the start of the filename", type=int)
    args = vars(parser.parse_args())
    verbose = not args['quiet']

    if args['testpath']==None:
        args['testpath'] = args['trainpath']

    # Build up lists of the training and testing WAV files:
    wavsfound = {'trainpath':{}, 'testpath':{}}
    for onepath in ['trainpath', 'testpath']:
        pattern = os.path.join(args[onepath], '*.wav')
        for wavpath in glob(pattern):
            if args['numchars'] != 0:
                label = os.path.basename(wavpath)[:args['numchars']]
            else:
                label = os.path.basename(wavpath).split(args['charsplit'])[0]
            shortwavpath = os.path.relpath(wavpath, args[onepath])
            wavsfound[onepath][shortwavpath] = label
        if len(wavsfound[onepath])==0:
            raise RuntimeError("Found no files using this pattern: %s" % pattern)
        if verbose:
            print("Class-labels and filenames to be used from %s:" % onepath)
            for wavpath,label in sorted(wavsfound[onepath].items()):
                print(" %s: \t %s" % (label, wavpath))

    if args['testpath'] != args['trainpath']:
        # Separate train-and-test collections
        ncorrect, ntotal, nclasses = trainAndTest(args['trainpath'], wavsfound['trainpath'], args['testpath'], wavsfound['testpath'])
        print("Got %i correct out of %i (trained on %i classes)" % (ncorrect, ntotal, nclasses))
    else:
        # This runs "stratified leave-one-out crossvalidation": test multiple times by leaving one-of-each-class out and training on the rest.
        # First we need to build a list of files grouped by each classlabel
        labelsinuse = sorted(list(set(wavsfound['trainpath'].values())))
        grouped = {label:[] for label in labelsinuse}
        for wavpath,label in wavsfound['trainpath'].items():
            grouped[label].append(wavpath)
        numfolds = min(len(collection) for collection in grouped.values())
        # Each "fold" will be a collection of one item of each label
        folds = [{wavpaths[index]:label for label,wavpaths in grouped.items()} for index in range(numfolds)]
        totcorrect, tottotal = (0,0)
        # Then we go through, each time training on all-but-one and testing on the one left out
        for index in range(numfolds):
            print("Fold %i of %i" % (index+1, numfolds))
            chosenfold = folds[index]
            alltherest = {}
            for whichfold, otherfold in enumerate(folds):
                if whichfold != index:
                    alltherest.update(otherfold)
            ncorrect, ntotal, nclasses = trainAndTest(args['trainpath'], alltherest, args['trainpath'], chosenfold)
            totcorrect += ncorrect
            tottotal   += ntotal
        print("Got %i correct out of %i (using stratified leave-one-out crossvalidation, %i folds)" % (totcorrect, tottotal, numfolds))

