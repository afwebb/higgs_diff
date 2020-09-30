'''
Inputs a text file listing input root files. 
Outputs a csv for each file with kinematics of various jets pairings to be used for top reconstruction training
Use match=1 if both jets are b-jets from top, match=0 otherwise
Write to csvFiles/top{2lSS 3l}/mc16{a,d,e}/<dsid>.csv (assumes input file is /path/mc16x/dsid.root)
Usage: python3.6 topReco.py <list of root files>
'''

import ROOT
from ROOT import TFile
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import sys
from numpy import unwrap, arange
import random
from joblib import Parallel, delayed
import multiprocessing 

def runReco(inf):
    ''' 
    load a root file, loop over each event. Create dataframe containing kinematics of different pairings of jets
    '''

    #Figure out which channel to use
    if '3l' in inf:
        channel='3l'
    elif '2lSS' in inf:
        channel='2lSS'
    else:
        print('Not sure which channel to use')
        exit()
        
    #Open the root file
    f = TFile.Open(inf)
    nom = f.Get('nominal')
    
    #initialize output dicts
    eventsTop = []
    
    #Loop over all entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
        
        nom.GetEntry(idx)
    
        #Separate jets into tops and others, add their indices
        topJets = []
        k = {}
        i, j = 0, 0
        while j<10:
            if len(nom.jet_pt)>i:
                if abs(nom.jet_eta[i])<3.5 and nom.jet_jvt[i]>0.59:                                                        
                    if abs(nom.jet_parents[i])==6 and abs(nom.jet_truthPartonLabel[i])==5:
                        topJets.append(1)
                    else:
                        topJets.append(0)
                    k['jet_pt_'+str(j)] = nom.jet_pt[i]
                    k['jet_eta_'+str(j)] = nom.jet_eta[i]
                    k['jet_phi_'+str(j)] = nom.jet_phi[i]
                    k['jet_e_'+str(j)] = nom.jet_e[i]
                    k['jet_DL1r_'+str(j)] = nom.jet_tagWeightBin_DL1r_Continuous[i]
                    j+=1
            else:
                topJets.append(0)
                k['jet_pt_'+str(j)] = 0
                k['jet_eta_'+str(j)] = 0
                k['jet_phi_'+str(j)] = 0
                k['jet_e_'+str(j)] = 0
                k['jet_DL1r_'+str(j)] = 0
                j+=1
            i+=1
            
        if sum(topJets)!=2: continue

        k['lep_pt_0']= nom.lep_Pt_0
        k['lep_eta_0']= nom.lep_Eta_0
        k['lep_phi_0']= nom.lep_Phi_0
        k['lep_e_0']= nom.lep_E_0

        k['lep_pt_1']= nom.lep_Pt_1                                                                                        
        k['lep_eta_1']= nom.lep_Eta_1                                                                               
        k['lep_phi_1']= nom.lep_Phi_1
        k['lep_e_1']= nom.lep_E_1

        if channel=='3l':
            k['lep_pt_2']= nom.lep_Pt_2
            k['lep_eta_2']= nom.lep_Eta_2
            k['lep_phi_2']= nom.lep_Phi_2
            k['lep_e_2']= nom.lep_E_2

            k['trilep_type'] = nom.trilep_type
        else:
            k['dilep_type'] = nom.dilep_type

        #k['nJets_OR_DL1r_85'] = nom.nJets_OR_DL1r_85
        #k['nJets_OR_DL1r_60'] = nom.nJets_OR_DL1r_60
        k['met'] = nom.met_met
        k['met_phi'] = nom.met_phi

        k['match'] = topJets
        eventsTop.append(k)

    # Convert to dataframe, shuffle entries
    dfTop = pd.DataFrame.from_dict(eventsTop)
    dfTop = shuffle(dfTop)

    #Write output to csv file
    outF = '/'.join(inf.split("/")[-2:]).replace('.root','.csv')
    if channel=='3l':
        dfTop.to_csv('csvFiles/top3l/'+outF, index=False, float_format='%.3f')
    else:
        dfTop.to_csv('csvFiles/top2lSS/'+outF, index=False, float_format='%.3f')        

#Run in parallel
linelist = [line.rstrip() for line in open(sys.argv[1])]
Parallel(n_jobs=10)(delayed(runReco)(inf) for inf in linelist)
