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
from dict_2l import dict2l, calc_phi
#from rootpy.io import root_open

#Read in list of files
inf = sys.argv[1]

#load xgb models

model = pickle.load(open("xgb_models/2l.dat", "rb"))
highModel = pickle.load(open("xgb_models/2lHigh.dat","rb"))
lowModel = pickle.load(open("xgb_models/2lLow.dat","rb"))

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
        
        k = dict2l(nom)
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

    inHigh = inDF#.drop(['pt_score'],axis=1)
    #inHigh = inHigh.drop(['bin_score'],axis=1)
    xgbMatHigh = xgb.DMatrix(inHigh, feature_names=list(inHigh))

    inLow = inDF#.drop(['pt_score'],axis=1)
    #inLow = inLow.drop(['bin_score'],axis=1)    
    xgbMatLow = xgb.DMatrix(inLow, feature_names=list(inLow))

    y_pred_2l = model.predict(xgbMat)
    y_pred_2lHigh = highModel.predict(xgbMatHigh)
    y_pred_2lLow = lowModel.predict(xgbMatLow)

    with root_open(inputPath, mode='a') as myfile:
        xgb_sigBkg_2l = np.asarray(y_pred_2l)
        xgb_sigBkg_2l.dtype = [('xgb_sigBkg_2l_2', 'float32')]
        xgb_sigBkg_2l.dtype.names = ['xgb_sigBkg_2l_2']
        root_numpy.array2tree(xgb_sigBkg_2l, tree=myfile.nominal)
        
        xgb_sigBkg_2lHigh = np.asarray(y_pred_2lHigh)
        xgb_sigBkg_2lHigh.dtype = [('xgb_sigBkg_2lHigh_2', 'float32')]
        xgb_sigBkg_2lHigh.dtype.names = ['xgb_sigBkg_2lHigh_2']
        root_numpy.array2tree(xgb_sigBkg_2lHigh, tree=myfile.nominal)

        xgb_sigBkg_2lLow = np.asarray(y_pred_2lLow)
        xgb_sigBkg_2lLow.dtype = [('xgb_sigBkg_2lLow_2', 'float32')]
        xgb_sigBkg_2lLow.dtype.names = ['xgb_sigBkg_2lLow_2']
        root_numpy.array2tree(xgb_sigBkg_2lLow, tree=myfile.nominal)

        myfile.write()
        myfile.Close()

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=20)(delayed(run_pred)(inFile) for inFile in linelist)

