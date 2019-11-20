import rootpy.io
from rootpy.tree import Tree
import sys
import pandas as pd
import math
from dict_3l import dict3l, calc_phi
from joblib import Parallel, delayed
import multiprocessing

inf = sys.argv[1]

listDSIDs = [x.rstrip() for x in open('../used_samples.txt', 'r')]

def run_csv(inFile):

    dsid = inFile.split('/')[-1]
    dsid = dsid.replace('.root', '')
    if dsid not in listDSIDs: return 0

    f = rootpy.io.root_open(inFile)
    
    outFile = "used_3lFiles/"+dsid
    if 'mc16a' in inFile:
        outFile = outFile+'a.csv'
    elif 'mc16d' in inFile:
        outFile = outFile+'d.csv'
    elif 'mc16e' in inFile:
        outFile = outFile+'e.csv'
                                                                        
    print(dsid, outFile)
    oldTree = f.get('nominal')
    #oldTree.SetBranchStatus("*",0)
    #for br in branch_list:
    #    oldTree.SetBranchStatus(br,1)

    events = []
    
    current = 0
    for e in oldTree:
        current+=1
        if current%10000==0:
            print(current)

        #if e.is3L==0: continue
        if e.trilep_type==0: continue
        if abs(e.total_charge)!=1: continue
        if e.nJets_OR_T<2: continue
        if e.nJets_OR_T_MV2c10_70<1: continue
        if e.lep_Pt_0<20e3 or e.lep_Pt_1<10e3 or e.lep_Pt_2<10e3: continue
        if e.lep_ID_0==-e.lep_ID_1 and abs(e.Mll01-91.2e3)<10e3: continue
        if e.lep_ID_0==-e.lep_ID_2 and abs(e.Mll02-91.2e3)<10e3: continue
        if e.MVA3lCERN_weight_ttH<-1: continue
        
        if '34587' in dsid or '34567' in dsid or '34634' in dsid:
            sig = 1
        else:
            sig = 0

        k = dict3l(e, sig)
        events.append(k)

    
    dFrame = pd.DataFrame(events)
    dFrame.to_csv(outFile, index=False)

linelist = [line.rstrip() for line in open(inf)]
print(linelist)
Parallel(n_jobs=20)(delayed(run_csv)(inFile) for inFile in linelist)
