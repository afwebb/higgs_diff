import xgboost as xgb
import ROOT
from ROOT import TFile
from rootpy.tree import Tree
from rootpy.vector import LorentzVector
from rootpy.io import root_open
from rootpy.tree import Tree, FloatCol, TreeModel
import root_numpy
import sys
import pickle
import math
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from dict_sigBkgBDT import sigBkgDict2l, sigBkgDict3l

#Read in list of files
inf = sys.argv[1]

#load xgb models

model_2lSS = pickle.load(open("models/2lSS.dat", "rb"))
model_2lSSHigh = pickle.load(open("models/2lSS_highPt.dat","rb"))
model_2lSSLow = pickle.load(open("models/2lSS_lowPt.dat","rb"))

model_3lF = pickle.load(open("models/3lF.dat", "rb"))
model_3lS = pickle.load(open("models/3lS.dat", "rb"))

model_3lFHigh = pickle.load(open("models/3lF_highPt.dat", "rb"))
model_3lSHigh = pickle.load(open("models/3lS_highPt.dat", "rb"))

model_3lFLow = pickle.load(open("models/3lF_lowPt.dat", "rb"))
model_3lSLow = pickle.load(open("models/3lS_lowPt.dat", "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

#create pt prediction dicts
def create_dict(nom, sigDict):
    current = 0

    events = []
    
    nEntries = nom.GetEntries()
    print(nEntries)
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))

        nom.GetEntry(idx)
        events.append( sigDict(nom) )

    return events

def addToRoot(inputPath, event_dict, model, name, toDrop=None):
    '''
    given an event array, make a prediction and add it to the root file
    '''

    inDF = pd.DataFrame(event_dict)                          
    if toDrop:
        inDF = inDF.drop([toDrop],axis=1)

    xgbMat = xgb.DMatrix(inDF, feature_names=list(inDF))
    y_pred = model.predict(xgbMat) 

    with root_open(inputPath, mode='a') as myfile:                                                                           
        y_pred = np.asarray(y_pred)                                            
        y_pred.dtype = [(name, 'float32')]                           
        y_pred.dtype.names = [name]                                                        
        root_numpy.array2tree(y_pred, tree=myfile.nominal)

        myfile.write() 
        myfile.Close()

#loop over file list, add prediction branches
def run_pred(inputPath):
    print(inputPath)
    f = TFile(inputPath, "READ")

    dsid = inputPath.split('/')[-1]
    dsid = dsid.replace('.root', '')
    print(dsid)
    
    nom = f.Get('nominal')
    if nom.GetEntries() == 0:
        return 0
    
    if '2lSS' in inputPath:
        event_dict = create_dict(nom, sigBkgDict2l)
        addToRoot(inputPath, event_dict, model_2lSS, 'sigBkg_2lSS', toDrop=None)
        addToRoot(inputPath, event_dict, model_2lSSHigh, 'sigBkg_2lSS_highPt', toDrop=None) #toDrop='recoHiggsPt_2lSS'
        addToRoot(inputPath, event_dict, model_2lSSLow, 'sigBkg_2lSS_lowPt', toDrop=None) #toDrop='recoHiggsPt_2lSS'

    else:
        event_dict = create_dict(nom, sigBkgDict3l)
        
        addToRoot(inputPath, event_dict, model_3lS, 'sigBkg_3lS', toDrop='recoHiggsPt_3lF')
        addToRoot(inputPath, event_dict, model_3lSHigh, 'sigBkg_3lS_highPt', toDrop='recoHiggsPt_3lF')
        addToRoot(inputPath, event_dict, model_3lSLow, 'sigBkg_3lS_lowPt', toDrop='recoHiggsPt_3lF')

        addToRoot(inputPath, event_dict, model_3lF, 'sigBkg_3lF', toDrop='recoHiggsPt_3lS')                         
        addToRoot(inputPath, event_dict, model_3lFHigh, 'sigBkg_3lF_highPt', toDrop='recoHiggsPt_3lS')               
        addToRoot(inputPath, event_dict, model_3lFLow, 'sigBkg_3lF_lowPt', toDrop='recoHiggsPt_3lS') 

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=20)(delayed(run_pred)(inFile) for inFile in linelist)

