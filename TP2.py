import librosa #https://librosa.org/    #sudo apt-get install -y ffmpeg (open mp3 files)
import librosa.display
import librosa.beat
import sounddevice as sd  #https://anaconda.org/conda-forge/python-sounddevice
from scipy import stats as sc
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    features = np.genfromtxt('Features - Audio MER\\top100_features.csv',dtype = str, delimiter = ',')
    features = np.delete(features,0,0) #apagar a primeira linha
    nomesMusicas = features[:, 0] #primeira coluna = nomes das musicas
    features = np.delete(features,0,1) #apagar a primeira coluna
    features = np.delete(features,100,1) #apagar a ultima coluna
    
    featuresF = features.astype(float)
    #features = np.nan_to_num(features, nan=0)
    featuresN = normalizar(featuresF) #normalizar a funcao

    #print(featuresN)
    #print(nomesMusicas)
    np.savetxt('featuresN.csv', featuresN, delimiter=',') #salvar matriz normalizada num excel
    extrairFeatures(nomesMusicas)

#correr a matriz linha a linha, obter o max e min de cada coluna e aplicar a formula caso n sejam iguais
def normalizar(matriz):
    matrizN = np.full_like(matriz,1)

    for i in range(len(matriz[0])):
        max = np.max(matriz[:, i])
        min = np.min(matriz[:, i])

        if (min == max):
            matrizN[:, i] = 0
        else:
            matrizN[:, i] = (0 + (matriz[:, i] - min)*(1 - 0))/(max - min)

    return matrizN
    
def stats(feat):
    return np.array([np.mean(feat), np.std(feat), sc.skew(feat), sc.kurtosis(feat), np.median(feat), np.max(feat), np.min(feat)])

def extrairFeatures(nomesMusicas):
    sr = 22050
    windowL = frameL = 92.88
    hopL = 23.22
    freqMin = 20
    freqMax = 11025
    
    nomesQ = np.array(['Q1','Q2','Q3','Q4'])

    statistics = np.zeros((900, 190))

    num_music = 0
    num_stats = 7
    num = 1
    for queryName in nomesQ:
        for music in os.listdir('MER_audio_taffc_dataset\\' + queryName):
            print('Music: '+str(num) + ' / 900')
            #nomeMus = nomeMus[1:-1]
            #caminho = 'MER_audio_taffc_dataset\\' + queryName +'\\' + nomeMus +'.mp3'
            #print(caminho)

            caminho = 'MER_audio_taffc_dataset\\' + queryName + '\\' + music

            y = librosa.load(caminho, sr=sr, mono = True)

            #features espectrais

            #mfcc
            mfcc = librosa.feature.mfcc(y = y[0], n_mfcc = 13, hop_length = int(hopL))
            
            #mfcc = mfcc[0, :]
            i = 0
            for i in range(mfcc.shape[0]):
                statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(mfcc[i, :])
                
            #print(i)
            i += 1
            #spectral centroid
            specCen = librosa.feature.spectral_centroid(y = y[0], hop_length = int(hopL))
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(specCen[0, :])
            
            #print(i)
            i += 1
            #spectral bandwidth
            specBand = librosa.feature.spectral_bandwidth(y=y[0])
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(specBand[0, :])

            #print(i)
            i += 1
            #spectral costrast
            spectralContr = librosa.feature.spectral_contrast(y=y[0])
            j = i
            for i in range(j, spectralContr.shape[0] + j):
                statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(spectralContr[i - j, :])
            

            #i = j
            #print(i)
            i += 1
            #spectral flatness
            spectralFlat = librosa.feature.spectral_flatness(y=y[0])
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(spectralFlat[0, :])

            #print(i)
            i += 1
            #spectral rolloff
            spectralRoll = librosa.feature.spectral_rolloff(y=y[0])
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(spectralRoll[0, :])

            #features temporais

            #print(i)
            i += 1
            #yin
            F0 = librosa.yin(y=y[0], fmin=freqMin, fmax=freqMax)
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(F0)

            #print(i)
            i += 1
            #RMS
            RMS = librosa.feature.rms(y=y[0])
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(RMS[0, :])

            #print(i)
            i += 1
            #zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y[0])
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(zero_crossing_rate[0, :])

            #outras features 

            #print(i)
            i += 1
            #tempo
            tempo = librosa.feature.rhythm.tempo(y=y[0])
            statistics[num_music, i * num_stats] = tempo[0]

            num_music = num_music + 1
            num+=1

    statisticsN = normalizar(statistics)

    np.savetxt('statisticsN.csv', statisticsN, delimiter=',')



if __name__ == '__main__':
    main()