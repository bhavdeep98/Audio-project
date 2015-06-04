# Audio-project
A software to determine which kind of environment we are in based on the sounds detected.

To run the wavconvert.py and mp3convert.py
You need to have pydub on your computer
For linux
    
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
    ./wavconvert.py Folder_Path # After givin the executable rights


