
import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
import warnings
import numpy as np
import matplotlib.pyplot as plt


def main():
    features = np.genfromtxt('Features_Audio_MER/top100_features.csv', delimiter = ',')

    features = np.nan_to_num(features, nan=0)

    print(features)

if __name__ == '__main__':
    main()