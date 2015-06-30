import os.path
import numpy as np
import argparse
from glob import glob
from scikits.audiolab import Sndfile
from scikits.audiolab import Format
from sklearn import svm
from MFCC import melScaling

#######################################################################
# some settings

framelen = 1024
#fs = 44100.0
fs = 16000.0
verbose = True

#######################################################################
# main class

class svm1:
    """
    USAGE EXAMPLE:
    In this hypothetical example we train on four audio files, labelled as either 'usa' or 'uk', and then test on a separate audio file:

    from gmmfl import svm1
            ll = gmm.eval(features)[0]
    model = svm1("wavs/training", {'karen01.wav':'usa', 'john01.wav':'uk', 'steve02.wav':'usa', 'joe03.wav':'uk'})
    model.classify('wavs/testing/hubert01.wav')
    """

    def __init__(self, wavfolder, trainingdata):
        """Initialise the classifier and train it on some WAV files.
        'wavfolder' is the base folder, to be prepended to all WAV paths.
        'trainingdata' is a dictionary of wavpath:label pairs."""

        self.mfccMaker = melScaling(int(fs), framelen/2, 40)
        self.mfccMaker.update()

        self.target={}
        self.labelarr=[]
        self.labelarr2={}
        self.labelarr3={}
        self.target2=[]
        i=0
        for wavpath in trainingdata:
            label = trainingdata[wavpath]
            if label not in self.labelarr2.values():
                print "    target classes :"+label
                self.labelarr.append(i)
                self.labelarr2[i]=label
                self.labelarr3[label]=i
                i=i+1
            else:
                self.labelarr.append(self.labelarr3[label])

        allfeatures = {wavpath:self.file_to_features(os.path.join(wavfolder, wavpath),trainingdata[wavpath]) for wavpath in trainingdata}

        #print("**********features  :",self.file_to_features(os.path.join(wavfolder,wavpath)))
        print("allfeatures :"+str(np.shape(allfeatures.values())))
        #print("target :",self.target2)


        print('target values', self.labelarr)
        print('target labels', self.labelarr2)
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

        # For each label's aggregated features, train a SVM and remember it
        #self.svms = {}
        self.X=[]
        for label, aggf in aggfeatures.items():
            if verbose: print("      Training a SVM for label %s, using data of shape %s" % (label, str(np.shape(aggf))))
            #self.svms[label] = svm.SVC()
            #print("aggf : ",aggf)
            if not np.any(self.X):
                self.X=aggf
            else:
                #print("X: ",np.shape(self.X))
                #print("aggf: ",np.shape(aggf))
                self.X=np.vstack((self.X,aggf))
            #self.svms[label].fit(aggf,self.target[label])
        #print("X :",self.X)
        self.clf=svm.SVC()
        #print("target : ",self.target2)
        self.clf.fit(self.X,self.target2)
        #if verbose: print("  Trained %i cslasses from %i input files" % (len(self.clf), len(trainingdata)))

    def __normalise(self, data):
        "Normalises data using the mean and stdev of the training data - so that everything is on a common scale."
        return (data - self.means) * self.invstds

    def classify(self, wavpath,label):
        "Specify the path to an audio file, and this returns the max-likelihood class, as a string label."
        features = self.__normalise(self.file_to_features(wavpath,label))
        # For each label SVM, find the best suitable class
        """bestlabel = ''
        for label, svm2 in self.svms.items():
            bestlabel = self.labelarr2[svm2.predict(features)[0]]
        """
        bestlabel = self.labelarr2[self.clf.predict(features)[0]]
        return bestlabel

    def file_to_features(self, wavpath,label):
        "Reads through a mono WAV file, converting each frame to the required features. Returns a 2D array."
        if verbose: print("Reading %s" % wavpath)
        if not os.path.isfile(wavpath): raise ValueError("path %s not found" % wavpath)
        sf = Sndfile(wavpath, "r")
        #if (sf.channels != 1) and verbose: print(" Sound file has multiple channels (%i) - channels will be mixed to mono." % sf.channels)
        if sf.samplerate != fs:         raise ValueError("wanted sample rate %g - got %g." % (fs, sf.samplerate))
        window = np.hamming(framelen)
        features = []
        tararr=[]
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

                framefeatures = melCepstrum

                features.append(framefeatures)
                tararr.append(self.labelarr3[label])
                self.target2.append(self.labelarr3[label])
                #print("features : ",framefeatures)
            except RuntimeError:
                break
        sf.close()
        if label not in self.target:
            self.target[label]=tararr
        else:
            self.target[label]=self.target[label]+tararr
        return np.array(features)

#######################################################################
def trainAndTest(trainpath, trainwavs, testpath, testwavs):
    "Handy function for evaluating your code: trains a model, tests it on wavs of known class. Returns (numcorrect, numtotal, numclasses)."
    print("TRAINING")
    model = svm1(trainpath, trainwavs)
    print("TESTING")
    ncorrect = 0
    for wavpath,label in testwavs.items():
        result = model.classify(os.path.join(testpath, wavpath),label)
        if verbose: print(" inferred: %s" % result)
        if result == label:
            ncorrect += 1
    return (ncorrect, len(testwavs), len(model.labelarr2))

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

