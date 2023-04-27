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
    
    #featuresN = guardaTop100Features()

    #statisticsN = extrairFeatures()
    

    featuresN = np.genfromtxt('featuresN.csv', dtype = float, delimiter = ',')

    statisticsN = np.genfromtxt('statisticsN.csv', dtype = float, delimiter = ',')

    similaridade(featuresN, statisticsN)
    statsEuc = np.genfromtxt('statsEuc.csv', dtype = float, delimiter = ',')
    statsMan = np.genfromtxt('statsMan.csv', dtype = float, delimiter = ',')
    statsCos = np.genfromtxt('statsCos.csv', dtype = float, delimiter = ',')
    featEuc = np.genfromtxt('featEuc.csv', dtype = float, delimiter = ',')
    featMan = np.genfromtxt('featMan.csv', dtype = float, delimiter = ',')
    featCos = np.genfromtxt('featCos.csv', dtype = float, delimiter = ',')

    rankingSimilaridade(statsEuc, statsMan, statsCos, featEuc, featMan, featCos)


def rankingSimilaridade(statsEuc, statsMan, statsCos, featEuc, featMan, featCos):

    quadrants = np.genfromtxt('MER_audio_taffc_dataset\\panda_dataset_taffc_annotations.csv',dtype = str, delimiter = ',')
    quadrants = np.delete(quadrants,0,0)

    querys = ['MT0000202045', 'MT0000379144', 'MT0000414517', 'MT0000956340']

    for i in querys:
        for j in range(quadrants.shape[0]):
            if quadrants[j, 0] == i:
                
                rankEucStatsidx = np.argsort(statsEuc[j, :])
                rankManStatsidx = np.argsort(statsMan[j, :])
                rankCosStatsidx = np.argsort(statsCos[j, :])
                rankEucFeatidx = np.argsort(featEuc[j, :])
                rankManFeatidx = np.argsort(featMan[j, :])
                rankCosFeatidx = np.argsort(featCos[j, :])

                #print(rankEucStatsidx)

                rankEucStats = np.zeros(20, dtype='U256')
                rankManStats = np.zeros(20, dtype='U256')   
                rankCosStats = np.zeros(20, dtype='U256')
                rankEucFeat = np.zeros(20, dtype='U256')
                rankManFeat = np.zeros(20, dtype='U256')              
                rankCosFeat = np.zeros(20, dtype='U256')


                #print(rankEucStatsidx[0 : 20])

                for k in range(20):
                    rankEucStats[k] = quadrants[rankEucStatsidx[k], 0]
                    rankManStats[k] = quadrants[rankManStatsidx[k], 0]
                    rankCosStats[k] = quadrants[rankCosStatsidx[k], 0]
                    rankEucFeat[k] = quadrants[rankEucFeatidx[k], 0]
                    rankManFeat[k] = quadrants[rankManFeatidx[k], 0]
                    rankCosFeat[k] = quadrants[rankCosFeatidx[k], 0]

                print("Query = " + i)

                print('Ranking: FMrosa, Euclidean')
                print(rankEucStats)

            
                break
            
                


def similaridade(featuresN, statisticsN):
    
    statsEuc = np.zeros((900, 900))
    statsMan = np.zeros((900, 900))
    statsCos = np.zeros((900, 900))
    featEuc = np.zeros((900, 900))
    featMan = np.zeros((900, 900))
    featCos = np.zeros((900, 900))

    k = 0
    for i in range(0, 900):
        for j in range(k, 900):
            if i == j:
                statsEuc[i, j] = -1
                statsMan[i, j] = -1
                statsCos[i, j] = -1
                featEuc[i, j] = -1
                featMan[i, j] = -1
                featCos[i, j] = -1
            else: 
                
                statsMan[i, j] = statsMan[j, i] =  np.linalg.norm(statisticsN[i, :] - statisticsN[j, :], ord=1)
                statsEuc[i, j] = statsEuc[j, i] = euclidiana(statisticsN, i, j)
                statsMan[i, j] = statsMan[j, i] = manhatten(statisticsN, i, j)
                statsCos[i, j] = statsCos[j, i] = cosseno(statisticsN, i, j)
                featEuc[i, j]  = featEuc[j, i] = euclidiana(featuresN, i, j)
                featMan[i, j]  = featMan[j, i] = manhatten(featuresN, i, j)
                featCos[i, j]  = featCos[j, i] = cosseno(featuresN, i, j)

        k += 1

    np.savetxt('statsEuc.csv', statsEuc, fmt="%lf", delimiter=',')
    np.savetxt('statsMan.csv', statsMan, fmt="%lf", delimiter=',')
    np.savetxt('statsCos.csv', statsCos, fmt="%lf", delimiter=',')
    np.savetxt('featEuc.csv', featEuc, fmt="%lf", delimiter=',')
    np.savetxt('featMan.csv', featMan, fmt="%lf", delimiter=',')
    np.savetxt('featCos.csv', featCos, fmt="%lf", delimiter=',')


def euclidiana(matriz, i, j):
    sum = 0
    for k in range(matriz.shape[1]):
        sum += np.power(matriz[i, k] - matriz[j, k], 2)

    sum = np.sqrt(sum)
    
    return sum
    

def manhatten(matriz, i, j):
    sum = 0
    for k in range(matriz.shape[1]):
        sum += np.abs(matriz[i, k] - matriz[j, k])

    return sum
    

def cosseno(matriz, i, j):
    sum = 0
    sum1 = 0
    sum2 = 0
    for k in range(matriz.shape[1]):
        sum += matriz[i, k] * matriz[j, k]
        sum1 += np.power(matriz[i, k], 2)
        sum2 += np.power(matriz[j, k], 2)

    sum1 = np.sqrt(sum1)
    sum2 = np.sqrt(sum2)

    sum = 1 - (sum/(sum1 * sum2))

    return sum

def guardaTop100Features():
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
    np.savetxt('featuresN.csv', featuresN, fmt="%lf", delimiter=',') #salvar matriz normalizada num excel
    return featuresN


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

def extrairFeatures():
    sr = 22050
    windowL = frameL = 92.88
    hopL = 23.22
    freqMin = 20
    freqMax = 11025

    quadrants = np.genfromtxt('MER_audio_taffc_dataset\\panda_dataset_taffc_annotations.csv',dtype = str, delimiter = ',')

    quadrants = np.delete(quadrants,0,0)
    #statistics = np.zeros((900, 190))
    statistics = np.zeros((900, 190))

    num_music = 0
    num_stats = 7
    num = 1

    for music in quadrants:
        print('Music: '+str(num) + ' / 900')
        #nomeMus = nomeMus[1:-1]
        #caminho = 'MER_audio_taffc_dataset\\' + queryName +'\\' + nomeMus +'.mp3'
        #print(caminho)

        caminho = 'MER_audio_taffc_dataset\\' + music[1] + '\\' + music[0] + '.mp3'
        #print(caminho)

        y = librosa.load(caminho, sr=sr, mono = True)

        #features espectrais

        #mfcc
        mfcc = librosa.feature.mfcc(y = y[0], n_mfcc = 13, hop_length = int(hopL))
        
        #mfcc = mfcc[0, :]
        i = 0
        for i in range(mfcc.shape[0]):
            statistics[num_music, i * num_stats : i * num_stats + num_stats] = stats(mfcc[i, :])
        
        #print(statistics[num_music, 0 * num_stats : 0 * num_stats + num_stats])
        #print(normalizar(statistics))
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

    #np.savetxt('teste.csv',normalizar(statistics), delimiter=',')
    
    np.savetxt('statistics.csv', statistics, fmt = "%lf", delimiter=',')
    statisticsN = normalizar(statistics)
    np.savetxt('statisticsN.csv', statisticsN, fmt = "%lf", delimiter=',')

    return statisticsN


if __name__ == '__main__':
    main()