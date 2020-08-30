''' 
Test the accuracey of the keras top matching algorithm for 2lSS  
Outputs the percentage of events the model correctly identifies both b-jets, and at least one b-jet
Usage: python3.6 testKerasHiggs.py <inputFile> <kerasModel> <kerasNormFactors>
'''

#import keras
from tensorflow.keras.models import load_model
import ROOT
import pandas as pd
import numpy as np
import sys
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector
import random
import pickle
from functionsW import WTopCombos, findBestTopKeras, findBestWTop

#Load in the input file
inf = sys.argv[1]
f = ROOT.TFile.Open(inf, "READ")
nom = f.Get('nominal')

#Load in the top matching keras model
WModel = load_model(sys.argv[2])
WModel.compile(loss="binary_crossentropy", optimizer='adam')

if '2lSS' in sys.argv[2]:
    channel = '2lSS'
    topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/keras_model_top2lSS.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/top2lSS_normFactors.npy")
elif '3l' in sys.argv[2]:
    channel = '3l'
    topModel = load_model("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/keras_model_top3l.h5")
    topNormFactors = np.load("/data_ceph/afwebb/higgs_diff/restructure/Wmatching/models/top3l_normFactors.npy")
else:
    print('Cannot determine which channel to use')
    exit

WNormFactors = np.load(sys.argv[3])

events = []
nEntries = nom.GetEntries()

nEvents=0
nCorrect=0

#Loop over each entry, add to events dict
for idx in range(nEntries):
    if idx%1000==0:
        print(str(idx)+'/'+str(nEntries))
    if idx==5000:
        break

    nom.GetEntry(idx)

    topRes = findBestTopKeras(nom, channel, topModel, topNormFactors)
    topIdx0, topIdx1 = topRes['bestComb']
    topScore = topRes['topScore']

    #Get dict of all possible jet combinations
    Wres = findBestWTop(nom, channel, WModel, WNormFactors, topIdx0, topIdx1, topScore)
    
    if len(Wres['truthLep'])!=1 or len(Wres['bestLep'])!=1:
        continue
    '''
    if channel=='2lSS':
        if Wres['WTopScore']<0.5 and Wres['truthLep'][0]==0:
            nCorrect+=1
        elif Wres['WTopScore']>0.5 and Wres['truthLep'][0]==1:
            nCorrect+=1
    elif channel=='3l':
        if Wres['WTopScore']<0.5 and Wres['truthLep'][0]==1:
            nCorrect+=1
        elif Wres['WTopScore']>0.5 and Wres['truthLep'][0]==2:
            nCorrect+=1 
    '''
    nEvents+=1
    if Wres['truthLep']==Wres['bestLep']:
        nCorrect+=1

outRes = open(f'models/testCorrect{channel}.txt', 'a')
outRes.write('\n\n')
outRes.write(f"{channel} W Top Matching Tested on {inf}\n")
outRes.write(f"Correct: {str(round(nCorrect/nEvents, 2))}\n\n")
    
