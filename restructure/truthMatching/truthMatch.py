''' 
Takes a root file as input, and matches the letpons and jets to the truth level parent (W, Higgs, top) 
Add lep_Parent_X and jet_parents variables
Usage: python3.6 truthMatch.py <input>
'''

import ROOT
from ROOT import TFile
import numpy as np
import numba as nb
import sys
import multiprocessing
from joblib import Parallel, delayed
import pickle
from array import array
    
def check_dR(pt, eta, phi, tpt, teta, tphi, dRcut, pTcut):
    ''' check if reco object matches a given truth object, requiring dR<0.1 and Pt within 10% '''
    return( abs(pt - tpt)/pt < pTcut and np.sqrt((eta - teta)**2 + (phi - tphi)**2) < dRcut )
    #return( np.sqrt((eta - teta)**2 + (phi - tphi)**2) < dRcut )

def get_parent(i, barcodes, pdgIds, parents):
    ''' Return the pdgId of the terminal parent of a truth particle '''
    parId = 0
    bar = barcodes[i]
    while bar in barcodes:
        parId = pdgIds[barcodes.index(bar)]
        try:
            bar = parents[barcodes.index(bar)][0]
        except:
            return parId
        
    return parId

def run_match(inFile, outFile):
#def truth_matching(nom):
    ''' loop over the nominal tree, returning a new TTree with truth matched kinematics '''

    if inFile==outFile: #Check if output and input have the same name
        print('input and output files are the same')
        return 0

    f = TFile(inFile, "READ") #open old file
    newFile = TFile(outFile, "RECREATE") #open new file
    nom = f.Get('nominal').CopyTree("total_leptons>=2 && total_charge!=0 && nJets_OR>=2 && nJets_OR_DL1r_70>=1") #copy old tree with selection. Include 2l and 3l events

    newTree = nom.CloneTree(0)#ROOT.TTree("nominal", "nominal")

    #initialize new branches
    lep_Parent_0 = array( 'i', [ 0 ] )
    lep_Parent_1 = array( 'i', [ 0 ] )
    lep_Parent_2 = array( 'i', [ 0 ] )
    jet_parents = ROOT.std.vector('int')() #np.array(1, dtype=np.int32)
    
    newTree.Branch('lep_Parent_0', lep_Parent_0, 'lep_Parent_0/I')
    newTree.Branch('lep_Parent_1', lep_Parent_1, 'lep_Parent_1/I')
    newTree.Branch('lep_Parent_2', lep_Parent_2, 'lep_Parent_2/I')
    newTree.Branch('jet_parents', jet_parents)#, 'jet_parents/I')

    #loop over entries
    nEntries = nom.GetEntries()
    for idx in range(nEntries):
        if idx%10000==0:
            print(str(idx)+'/'+str(nEntries))
        
        nom.GetEntry(idx)

        #reset new branches to 0
        lep_Parent_0[0] = 0
        lep_Parent_1[0] = 0
        lep_Parent_2[0] = 0
        jet_parents.clear()
        
        #Make truth vectors readable in python
        truth_barcode = [x for x in nom.m_truth_barcode]
        truth_pdgId = [x for x in nom.m_truth_pdgId]
        truth_parents = np.asarray(nom.m_truth_parents) # is an array of arrays

        if len(truth_barcode)!=len(truth_parents):
            print('diff parent, barcode lengths, skipping')
            continue
            
        # Get the parent pdgId for each truth matched lepton                                                      
        lepParents = [0 for x in range(nom.total_leptons)]
        for i, (tPt, tEta, tPhi, tFlav) in enumerate(zip(nom.m_truth_pt, nom.m_truth_eta, 
                                                         nom.m_truth_phi, nom.m_truth_pdgId)):
            if abs(nom.lep_ID_0)==abs(tFlav):
                if check_dR(nom.lep_Pt_0, nom.lep_Eta_0, nom.lep_Phi_0, tPt, tEta, tPhi, 0.1, 0.8):
                    lep_Parent_0[0] = get_parent(i, truth_barcode, truth_pdgId, truth_parents)
                    break
        
        for i, (tPt, tEta, tPhi, tFlav) in enumerate(zip(nom.m_truth_pt, nom.m_truth_eta, 
                                                  nom.m_truth_phi, nom.m_truth_pdgId)):
            if abs(nom.lep_ID_1)==abs(tFlav):
                if check_dR(nom.lep_Pt_1, nom.lep_Eta_1, nom.lep_Phi_1, tPt, tEta, tPhi, 0.1, 0.8):
                    lep_Parent_1[0] = get_parent(i, truth_barcode, truth_pdgId, truth_parents)
                    break
        
        if nom.total_leptons==3:
            for i, (tPt, tEta, tPhi, tFlav) in enumerate(zip(nom.m_truth_pt, nom.m_truth_eta, 
                                                             nom.m_truth_phi, nom.m_truth_pdgId)):
                if abs(nom.lep_ID_2)==abs(tFlav):
                    if check_dR(nom.lep_Pt_2, nom.lep_Eta_2, nom.lep_Phi_2, tPt, tEta, tPhi, 0.1, 0.8):
                        lep_Parent_2[0] = get_parent(i, truth_barcode, truth_pdgId, truth_parents)          
                        break

        # Get the pdgId of the parent of each reco jet
        for j, (jPt, jEta, jPhi, jFlav) in enumerate(zip(nom.jet_pt, nom.jet_eta, nom.jet_phi, nom.jet_truthPartonLabel)):
            match = 0
            for i, (tPt, tEta, tPhi, tFlav) in enumerate(zip(nom.m_truth_pt, nom.m_truth_eta, 
                                                             nom.m_truth_phi, nom.m_truth_pdgId)):
                if abs(tFlav)==jFlav:
                    if check_dR(jPt, jEta, jPhi, tPt, tEta, tPhi, 0.3, 0.25):
                        match = get_parent(i, truth_barcode, truth_pdgId, truth_parents)
                        break
            jet_parents.push_back(match)
        
        newTree.Fill()
        
    newTree.Write()
    newFile.Close()

inFile = sys.argv[1]
#Tried multiprocessing, didn't really work with root objects
#linelist = [line.rstrip() for line in open(inf)]
#Parallel(n_jobs=10)(delayed(run_match)(inFile, inFile.split("/")[-1]) for inFile in linelist)
#[run_match(inFile, 'outRoot/'+'/'.join(inFile.split("/")[-2:])) for inFile in linelist]
run_match(inFile, 'outRoot/'+'/'.join(inFile.split("/")[-2:]))
