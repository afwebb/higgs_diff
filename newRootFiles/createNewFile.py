import rootpy.io
import uproot
from rootpy import stl
from rootpy.io import File, root_open
from rootpy.tree import Tree,TreeModel, FloatCol, IntCol
import sys
from random import gauss
import math
from math import sqrt
from numpy import unwrap
from numpy import arange
from rootpy.vector import LorentzVector

inputFile = sys.argv[1]
outputFile = sys.argv[2]

inF = uproot.open(inputFile)
nom=inF.get('nominal')

la=inF['nominal'].lazyarrays(['m_truth*','dilep_type','trilep_type','quadlep_type','higgs*','lep_Pt*','lep_Eta_*', 'lep_E_*','total_charge',
                            'total_leptons', 'lep_Phi*','lep_ID*','lep_Index*', 'm_track_jet*', 'm_jet*','selected_jets', 
                            'm_truth_jet_pt', 'm_truth_jet_eta', 'm_truth_jet_phi', 'm_truth_jet_m','nJets_OR_T','nJets_OR_T_MV2c10_70',
                            'MET_RefFinal_et', 'MET_RefFinal_phi'])

from collections import namedtuple
parttype = namedtuple('parttype', ['barcode', 'pdgid', 'status', 'eta', 'phi', 'pt', 'e', 'parents', 'children'])

def make_partdict(la, idx):
    rv = dict(zip(la[b'm_truth_barcode'][idx],
                 (parttype(*_2) for
                  _2 in zip(la[b'm_truth_barcode'][idx],
                            la[b'm_truth_pdgId'][idx],
                            la[b'm_truth_status'][idx],
                            la[b'm_truth_eta'][idx],
                            la[b'm_truth_phi'][idx],
                            la[b'm_truth_pt'][idx],
                            la[b'm_truth_e'][idx],
                            la[b'm_truth_parents'][idx],
                            la[b'm_truth_children'][idx])
                 )
                 ))
    return rv

def drCheck(eta, phi, truth_eta, truth_phi, cut):
    dr = sqrt( (phi-truth_phi)**2 + (eta-truth_eta)**2 )
    return dr < cut

def lepMatch(lep_eta, lep_phi, lep_id, c):
    lepIDs = []
    par = 0
    for x in c:
        if abs(c[x].pdgid) in [11, 13]: lepIDs.append(x)

    for a in lepIDs:
        if abs(c[a].pdgid) == abs(lep_id):
            if drCheck(lep_eta, lep_phi, c[a].eta, c[a].phi, 0.01):
                p = c[a].parents[0]
                terminal = False
                while not terminal:
                    if p in c:
                        a = p
                        try:
                            p = c[a].parents[0]
                        except:
                            terminal = True
                        else: terminal = True
                par = c[a].pdgid
    return par

def higgsJets(c):
    higgsID = 0
    for x in c:
        if c[x].pdgid==25:
            higgsID = x

    jetCands = []
    jetTest = []
    lepTest = []
    Ws = c[higgsID].children
    for w in Ws:
        try:
            if 24 in [abs(c[x].pdgid) for x in c[w].children]:
                childCand = []
                for wChild in c[w].children:
                    if c[wChild].pdgid in [-24, 24]:
                        for x in c[wChild].children:
                            childCand.append(x)
            else:
                childCand = c[w].children
        except:
            childCand = c[w].children

        for child in childCand:
            if child in c:
                ch = c[child]
            else:
                continue
            if abs(ch.pdgid) in [11, 13]:
                lepTest.append(child)
            elif abs(ch.pdgid) in range(1,5):
                jetTest.append(child)

    return lepTest, jetTest

def topJets(c):
    topID = []
    for x in c:
        if abs(c[x].pdgid)==6: topID.append(x)

    jetCands = []
    lepTest = []

    Ws = [*c[topID[0]].children, *c[topID[1]].children]

    for w in Ws:
        if abs(c[w].pdgid)==5:
            jetCands.append(w)
        else:
            try:
                if 24 in [abs(c[x].pdgid) for x in c[w].children]:
                    childCand = []
                    for wChild in c[w].children:
                        if c[wChild].pdgid in [-24, 24]:
                            for x in c[wChild].children:
                                childCand.append(x)
                else:
                    childCand = c[w].children
            except:
                childCand = c[w].children

            for child in childCand:
                if child in c:
                    ch = c[child]
                else:
                    continue
                if abs(ch.pdgid) in [11, 13]:
                    lepTest.append(child)
        
    return lepTest, jetCands

outF = root_open(outputFile, 'recreate')

class Model(TreeModel):

    #Selection branches
    higgsDecayMode = FloatCol()
    total_leptons = FloatCol()
    total_charge = FloatCol()
    dilep_type = FloatCol()
    trilep_type = FloatCol()
    quadlep_type = FloatCol()

    nJets = FloatCol()
    nJets_MV2c10_70 = FloatCol()

    higgs_pt = FloatCol()

    #Jet Branches
    jet_pt = stl.vector('float')
    jet_eta = stl.vector('float')
    jet_phi = stl.vector('float')
    jet_E = stl.vector('float')
    jet_MV2c10 = stl.vector('float')
    jet_parent = stl.vector('float')
    jet_flavor = stl.vector('float')

    jet_jvt = stl.vector('float')
    jet_numTrk = stl.vector('float')

    track_jet_pt = stl.vector('float')
    track_jet_eta = stl.vector('float')
    track_jet_phi = stl.vector('float')
    track_jet_m = stl.vector('float')
    track_jet_MV2c10 = stl.vector('float')
    track_jet_parent = stl.vector('float')
    track_jet_flavor = stl.vector('float')

    truth_jet_pt = stl.vector('float')
    truth_jet_eta = stl.vector('float')
    truth_jet_phi = stl.vector('float')
    truth_jet_E = stl.vector('float')
    truth_jet_parent = stl.vector('float')
    truth_jet_flavor = stl.vector('float')

    #Lep Branches
    lep_pt = stl.vector('float')
    lep_eta = stl.vector('float')
    lep_phi = stl.vector('float')
    lep_E = stl.vector('float')
    lep_parent = stl.vector('float')
    lep_flavor = stl.vector('float')

    truth_lep_pt = stl.vector('float')
    truth_lep_eta = stl.vector('float')
    truth_lep_phi = stl.vector('float')
    truth_lep_E = stl.vector('float')
    truth_lep_parent = stl.vector('float')
    truth_lep_flavor = stl.vector('float')

    #MET
    met = FloatCol()
    met_phi = FloatCol()

tree = Tree('nominal', model=Model)

for idx in range( len(la[b'nJets_OR_T']) ):

    if idx%10000==0:
        print(idx)
    #if idx==5000:                                                                                                             
    #    break                                                                                                                 

    #Event selection
    #if la[b'higgsDecayMode'][idx] != 3: continue
    if la[b'total_leptons'][idx] < 2: continue
    #if la[b'dilep_type'][idx] < 1: continue
    #if la[b'total_charge'][idx] == 0: continue
    if la[b'nJets_OR_T_MV2c10_70'][idx] < 1: continue
    if la[b'nJets_OR_T'][idx] < 2: continue 

    #Clear vectors
    tree.jet_pt.clear()
    tree.jet_eta.clear()
    tree.jet_phi.clear()
    tree.jet_E.clear()
    tree.jet_MV2c10.clear()
    tree.jet_parent.clear()
    tree.jet_flavor.clear()

    tree.jet_jvt.clear()
    tree.jet_numTrk.clear()

    tree.lep_pt.clear()
    tree.lep_eta.clear()
    tree.lep_phi.clear()
    tree.lep_E.clear()
    tree.lep_parent.clear()
    tree.lep_flavor.clear()
    '''
    tree.track_jet_pt.clear()
    tree.track_jet_eta.clear()
    tree.track_jet_phi.clear()
    tree.track_jet_m.clear()
    tree.track_jet_MV2c10.clear()
    tree.track_jet_parent.clear()
    tree.track_jet_flavor.clear()
    '''
    tree.truth_jet_pt.clear()
    tree.truth_jet_eta.clear()
    tree.truth_jet_phi.clear()
    tree.truth_jet_E.clear()
    tree.truth_jet_parent.clear()
    tree.truth_jet_flavor.clear()

    tree.truth_lep_pt.clear()
    tree.truth_lep_eta.clear()
    tree.truth_lep_phi.clear()
    tree.truth_lep_E.clear()
    tree.truth_lep_parent.clear()
    tree.truth_lep_flavor.clear()

    #Add Flat Branches
    tree.higgs_pt = la[b'higgs_pt'][idx]

    tree.higgsDecayMode = la[b'higgsDecayMode'][idx] 
    tree.total_leptons = la[b'total_leptons'][idx]
    tree.total_charge = la[b'total_charge'][idx]
    tree.dilep_type = la[b'dilep_type'][idx]
    tree.trilep_type = la[b'trilep_type'][idx]
    tree.quadlep_type = la[b'quadlep_type'][idx]
    
    tree.nJets = la[b'nJets_OR_T'][idx]
    tree.nJets_MV2c10_70 = la[b'nJets_OR_T_MV2c10_70'][idx]

    tree.met = la[b'MET_RefFinal_et'][idx]
    tree.met_phi = la[b'MET_RefFinal_phi'][idx]

    truth_dict = make_partdict(la, idx)

    hLep, hJets = higgsJets(truth_dict)
    tLep, tJets = topJets(truth_dict)

    for x in hLep:
        tree.truth_lep_pt.push_back( truth_dict[x].pt )
        tree.truth_lep_eta.push_back( truth_dict[x].eta )
        tree.truth_lep_phi.push_back( truth_dict[x].phi )
        tree.truth_lep_E.push_back( truth_dict[x].e )
        tree.truth_lep_parent.push_back( 25 )
        tree.truth_lep_flavor.push_back( truth_dict[x].pdgid )

    for x in hLep:
        tree.truth_lep_pt.push_back( truth_dict[x].pt )
        tree.truth_lep_eta.push_back( truth_dict[x].eta )
        tree.truth_lep_phi.push_back( truth_dict[x].phi )
        tree.truth_lep_E.push_back( truth_dict[x].e )
        tree.truth_lep_parent.push_back( 6 )
        tree.truth_lep_flavor.push_back( truth_dict[x].pdgid )

    for x in hJets:
        tree.truth_jet_pt.push_back( truth_dict[x].pt )
        tree.truth_jet_eta.push_back( truth_dict[x].eta )
        tree.truth_jet_phi.push_back( truth_dict[x].phi )
        tree.truth_jet_E.push_back( truth_dict[x].e )
        tree.truth_jet_parent.push_back( 25 )
        tree.truth_jet_flavor.push_back( truth_dict[x].pdgid )

    for x in hJets:
        tree.truth_jet_pt.push_back( truth_dict[x].pt )
        tree.truth_jet_eta.push_back( truth_dict[x].eta )
        tree.truth_jet_phi.push_back( truth_dict[x].phi )
        tree.truth_jet_E.push_back( truth_dict[x].e )
        tree.truth_jet_parent.push_back( 6 )
        tree.truth_jet_flavor.push_back( truth_dict[x].pdgid )

    #Fill Lep Branches
    tree.lep_pt.push_back( la[b'lep_Pt_0'][idx] )
    tree.lep_eta.push_back( la[b'lep_Eta_0'][idx] )
    tree.lep_phi.push_back( la[b'lep_Phi_0'][idx] )
    tree.lep_E.push_back( la[b'lep_E_0'][idx] )
    tree.lep_flavor.push_back( la[b'lep_ID_0'][idx] )
    #tree.lep_parent.push_back( lepMatch(la[b'lep_Eta_0'][idx], la[b'lep_Phi_0'][idx], la[b'lep_ID_0'][idx], truth_dict) )
    lepPar0 = 0
    for j in hLep:
        dr = sqrt(unwrap([ la[b'lep_Phi_0'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_0'][idx] - truth_dict[j].eta)**2)
        if dr<0.1 and abs(la[b'lep_ID_0'][idx])==abs(truth_dict[j].pdgid):
            lepPar0 = 25
    for j in tLep:
        dr = sqrt(unwrap([ la[b'lep_Phi_0'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_0'][idx] - truth_dict[j].eta)**2)
        if dr<0.1 and abs(la[b'lep_ID_0'][idx])==abs(truth_dict[j].pdgid):
            lepPar0 = 6

    tree.lep_parent.push_back( lepPar0 )

    if la[b'total_leptons'][idx]>1:
        tree.lep_pt.push_back( la[b'lep_Pt_1'][idx] )
        tree.lep_eta.push_back( la[b'lep_Eta_1'][idx] )
        tree.lep_phi.push_back( la[b'lep_Phi_1'][idx] )
        tree.lep_E.push_back( la[b'lep_E_1'][idx] )
        tree.lep_flavor.push_back( la[b'lep_ID_1'][idx] )
    #tree.lep_parent.push_back( lepMatch(la[b'lep_Eta_1'][idx], la[b'lep_Phi_1'][idx], la[b'lep_ID_1'][idx], truth_dict) )
        lepPar1 = 0
        for j in hLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_1'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_1'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_1'][idx])==abs(truth_dict[j].pdgid):
                lepPar1 = 25
        for j in tLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_1'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_1'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_1'][idx])==abs(truth_dict[j].pdgid):
                lepPar1 = 6
        tree.lep_parent.push_back( lepPar1 )

    if la[b'total_leptons'][idx]>2:
        tree.lep_pt.push_back( la[b'lep_Pt_2'][idx] )
        tree.lep_eta.push_back( la[b'lep_Eta_2'][idx] )
        tree.lep_phi.push_back( la[b'lep_Phi_2'][idx] )
        tree.lep_E.push_back( la[b'lep_E_2'][idx] )
        tree.lep_flavor.push_back( la[b'lep_ID_2'][idx] )

        lepPar2 = 0
        for j in hLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_2'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_2'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_2'][idx])==abs(truth_dict[j].pdgid):
                    lepPar2 = 25
        for j in tLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_2'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_2'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_2'][idx])==abs(truth_dict[j].pdgid):
                lepPar2 = 6
        tree.lep_parent.push_back( lepPar2 )

    if la[b'total_leptons'][idx]>3:
        tree.lep_pt.push_back( la[b'lep_Pt_3'][idx] )
        tree.lep_eta.push_back( la[b'lep_Eta_3'][idx] )
        tree.lep_phi.push_back( la[b'lep_Phi_3'][idx] )
        tree.lep_E.push_back( la[b'lep_E_3'][idx] )
        tree.lep_flavor.push_back( la[b'lep_ID_3'][idx] )

        lepPar3 = 0
        for j in hLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_3'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_3'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_3'][idx])==abs(truth_dict[j].pdgid):
                lepPar3 = 25
        for j in tLep:
            dr = sqrt(unwrap([ la[b'lep_Phi_3'][idx] - truth_dict[j].phi])**2+( la[b'lep_Eta_3'][idx] - truth_dict[j].eta)**2)
            if dr<0.1 and abs(la[b'lep_ID_3'][idx])==abs(truth_dict[j].pdgid):
                lepPar3 = 6
        tree.lep_parent.push_back( lepPar3 )

    #Fill Jet Branches
    for i in range(len(la[b'm_jet_pt'][idx])):
        tree.jet_pt.push_back( la[b'm_jet_pt'][idx][i] )
        tree.jet_eta.push_back( la[b'm_jet_eta'][idx][i] )
        tree.jet_phi.push_back( la[b'm_jet_phi'][idx][i] )
        tree.jet_E.push_back( la[b'm_jet_E'][idx][i] )
        tree.jet_flavor.push_back( la[b'm_jet_flavor_truth_label_ghost'][idx][i] )
        tree.jet_MV2c10.push_back( la[b'm_jet_flavor_weight_MV2c10'][idx][i] )
        
        tree.jet_jvt.push_back( la[b'm_jet_jvt'][idx][i] )
        tree.jet_numTrk.push_back( la[b'm_jet_numTrk'][idx][i] )
        
        jet_flav = la[b'm_jet_flavor_truth_label_ghost'][idx][i] 

        jetPar = 0
        for j in hJets:
            dr = sqrt(unwrap([ la[b'm_jet_phi'][idx][i] - truth_dict[j].phi])**2+( la[b'm_jet_eta'][idx][i] - truth_dict[j].eta)**2)
            if dr<0.3 and jet_flav==abs(truth_dict[j].pdgid):
                jetPar = 25
        for j in tJets:
            dr = sqrt(unwrap([ la[b'm_jet_phi'][idx][i] - truth_dict[j].phi])**2+( la[b'm_jet_eta'][idx][i] - truth_dict[j].eta)**2)
            if dr<0.3 and jet_flav==abs(truth_dict[j].pdgid):
                jetPar = 6
                
        tree.jet_parent.push_back( jetPar )

        #Fill Jet Branches                                                                                                          
    '''
    for i in range(len(la[b'm_track_jet_pt'][idx])):
        tree.track_jet_pt.push_back( la[b'm_track_jet_pt'][idx][i] )
        tree.track_jet_eta.push_back( la[b'm_track_jet_eta'][idx][i] )
        tree.track_jet_phi.push_back( la[b'm_track_jet_phi'][idx][i] )
        tree.track_jet_m.push_back( la[b'm_track_jet_m'][idx][i] )
        tree.track_jet_flavor.push_back( la[b'm_track_jet_PartonTruthLabelID'][idx][i] )
        tree.track_jet_MV2c10.push_back( la[b'm_track_jet_flavor_weight_MV2c10'][idx][i] )
        track_jet_flav = la[b'm_track_jet_PartonTruthLabelID'][idx][i]

        track_jetPar = 0
        for j in hJets:
            dr = sqrt(unwrap([ la[b'm_track_jet_phi'][idx][i] - truth_dict[j].phi])**2+( la[b'm_track_jet_eta'][idx][i] - truth_dict[j].eta)**2)
            if dr<0.3 and track_jet_flav==abs(truth_dict[j].pdgid):
                track_jetPar = 25
        for j in tJets:
            dr = sqrt(unwrap([ la[b'm_track_jet_phi'][idx][i] - truth_dict[j].phi])**2+( la[b'm_track_jet_eta'][idx][i] - truth_dict[j].eta)**2)
            if dr<0.3 and track_jet_flav==abs(truth_dict[j].pdgid):
                track_jetPar = 6

        tree.track_jet_parent.push_back( track_jetPar )
    '''
    tree.fill()

tree.write()

outF.close()
