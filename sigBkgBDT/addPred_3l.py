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
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
import dict_3l
from dict_3l import dict3l, calc_phi
#from rootpy.io import root_open

#Read in list of files
inf = sys.argv[1]

#load xgb models

#model_3l = pickle.load(open("xgb_models/3l.dat", "rb"))

model_3lF = pickle.load(open("xgb_models/3lF.dat", "rb"))
model_3lS = pickle.load(open("xgb_models/3lS.dat", "rb"))

model_3lFHigh = pickle.load(open("xgb_models/3lFHigh.dat", "rb"))
model_3lSHigh = pickle.load(open("xgb_models/3lSHigh.dat", "rb"))

model_3lFLow = pickle.load(open("xgb_models/3lFLow.dat", "rb"))
model_3lSLow = pickle.load(open("xgb_models/3lSLow.dat", "rb"))

def calc_phi(phi_0, new_phi):
    new_phi = new_phi-phi_0
    if new_phi>math.pi:
        new_phi = new_phi - 2*math.pi
    if new_phi<-math.pi:
        new_phi = new_phi + 2*math.pi
    return new_phi

#create pt prediction dicts
def create_dict(nom):
    current = 0

    events = []
    
    nEntries = nom.GetEntries()
    print(nEntries)
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))

        nom.GetEntry(idx)

        k = dict3l(nom)
        
        events.append(k)

    return events

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
    
    event_dict = create_dict(nom)
    inDF = pd.DataFrame(event_dict)
    xgbMat = xgb.DMatrix(inDF, feature_names=list(inDF))

    in3lF = inDF.drop(['pt_score_3lS'],axis=1)
    #in3lF = in3lF.drop(['bin_score_3lS'],axis=1)

    in3lS = inDF.drop(['pt_score_3lF'],axis=1)
    #in3lS = in3lS.drop(['bin_score_3lF'],axis=1)

    xgbMat3lF = xgb.DMatrix(in3lF, feature_names=list(in3lF))
    xgbMat3lS = xgb.DMatrix(in3lS, feature_names=list(in3lS))

    #y_pred_3l = model_3l.predict(xgbMat)

    y_pred_3lF = model_3lF.predict(xgbMat3lF)
    y_pred_3lS = model_3lS.predict(xgbMat3lS)

    y_pred_3lFHigh = model_3lFHigh.predict(xgbMat3lF)
    y_pred_3lSHigh = model_3lSHigh.predict(xgbMat3lS)

    y_pred_3lFLow = model_3lFLow.predict(xgbMat3lF)
    y_pred_3lSLow = model_3lSLow.predict(xgbMat3lS)


    with root_open(inputPath, mode='a') as myfile:
        #xgb_sigBkg_3l = np.asarray(y_pred_3l)
        #xgb_sigBkg_3l.dtype = [('xgb_sigBkg_3l_2', 'float32')]
        #xgb_sigBkg_3l.dtype.names = ['xgb_sigBkg_3l_2']
        #root_numpy.array2tree(xgb_sigBkg_3l, tree=myfile.nominal)

        xgb_sigBkg_3lF = np.asarray(y_pred_3lF)
        xgb_sigBkg_3lF.dtype = [('xgb_sigBkg_3lF_2', 'float32')]
        xgb_sigBkg_3lF.dtype.names = ['xgb_sigBkg_3lF_2']
        root_numpy.array2tree(xgb_sigBkg_3lF, tree=myfile.nominal)
        
        xgb_sigBkg_3lS = np.asarray(y_pred_3lS)
        xgb_sigBkg_3lS.dtype = [('xgb_sigBkg_3lS_2', 'float32')]
        xgb_sigBkg_3lS.dtype.names = ['xgb_sigBkg_3lS_2']
        root_numpy.array2tree(xgb_sigBkg_3lS, tree=myfile.nominal)

        xgb_sigBkg_3lFHigh = np.asarray(y_pred_3lFHigh)
        xgb_sigBkg_3lFHigh.dtype = [('xgb_sigBkg_3lFHigh_2', 'float32')]
        xgb_sigBkg_3lFHigh.dtype.names = ['xgb_sigBkg_3lFHigh_2']
        root_numpy.array2tree(xgb_sigBkg_3lFHigh, tree=myfile.nominal)

        xgb_sigBkg_3lSHigh = np.asarray(y_pred_3lSHigh)
        xgb_sigBkg_3lSHigh.dtype = [('xgb_sigBkg_3lSHigh_2', 'float32')]
        xgb_sigBkg_3lSHigh.dtype.names = ['xgb_sigBkg_3lSHigh_2']
        root_numpy.array2tree(xgb_sigBkg_3lSHigh, tree=myfile.nominal)

        xgb_sigBkg_3lFLow = np.asarray(y_pred_3lFLow)
        xgb_sigBkg_3lFLow.dtype = [('xgb_sigBkg_3lFLow_2', 'float32')]
        xgb_sigBkg_3lFLow.dtype.names = ['xgb_sigBkg_3lFLow_2']
        root_numpy.array2tree(xgb_sigBkg_3lFLow, tree=myfile.nominal)

        xgb_sigBkg_3lSLow = np.asarray(y_pred_3lSLow)
        xgb_sigBkg_3lSLow.dtype = [('xgb_sigBkg_3lSLow_2', 'float32')]
        xgb_sigBkg_3lSLow.dtype.names = ['xgb_sigBkg_3lSLow_2']
        root_numpy.array2tree(xgb_sigBkg_3lSLow, tree=myfile.nominal)

        myfile.write()
        myfile.Close()

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=10)(delayed(run_pred)(inFile) for inFile in linelist)

