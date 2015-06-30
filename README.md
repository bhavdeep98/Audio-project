# Audio-project
A software to determine which kind of environment we are in based on the sounds detected.

wavconvert.py:
This program is to  convert all type of files into the wav files if need be. 
Dependency:
	pydub
		pip install pydub
		or
		git clone https://github.com/jiaaro/pydub.git

    		cd pydub
    		python setup.py build
    		sudo python setup.py install # Need to to root to run this command so don't forget sudo

	    	PYTHONPATH="#Path to pydub libraries"
    		export $PYTHONPATH
To run
    	python wavconvert.py Folder_Path
    	or
	chmod 755 wavconvert,py
    	./wavconvert.py Folder_Path #All the files if possible will be converted to wav format

gmmfl.py:
The Gausian mixture Model implementation for our problem
Dependency:
	scikit.audiolab:
		Download source code from:
		https://pypi.python.org/pypi/scikits.audiolab/
		To extract:
		tar xvzf filename # here z option is for gzip file if you download a bzip2 file from somewhere the change x to j
		
