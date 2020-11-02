#import rootpy.io
#from rootpy.tree import Tree
import ROOT
import sys
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
from dict_sigBkdBDT import sigBkgDict2l, sigBkgDict3l

import math

inf = sys.argv[1]
#listDSIDs = [x.rstrip() for x in open('../used_samples.txt', 'r')]

def run_csv(inFile):

    if '2lSS' in inFile:
        sigDict = sigBkgDict2l
    else:
        sigDict = sigBkgDict3l

    dsid = inFile.split('/')[-1]
    dsid = dsid.replace('.root', '')

    f = ROOT.TFile(inFile)#rootpy.io.root_open(inFile)
    outFile = '/'.join(inFile.split("/")[-2:]).replace('.root','.csv')
    print(dsid, outFile)
    nom = f.Get('nominal')
    
    events = []

    current = 0
    for idx in range(nom.GetEntries()):
        current+=1
        if current%10000==0:
            print(current)

        nom.GetEntry(idx)
       
        if '34587' in dsid or '34567' in dsid or '34634' in dsid:
            sig = 1
        else:
            sig = 0

        k = sigDict(nom, sig)

        events.append(k)

    
    dFrame = pd.DataFrame(events)
    dFrame.to_csv('csvFiles/'+outFile, index=False)

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=30)(delayed(run_csv)(inFile) for inFile in linelist)
