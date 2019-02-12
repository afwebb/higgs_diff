import rootpy.io
from rootpy.tree import Tree
import ROOT
import sys

inFiles = sys.argv[1:]

branch_list = ['higgs_pt', 'is2LSS0Tau', 'lep_Pt_0', 'lep_Eta_0', 'lep_Phi_0', 'lep_Pt_1', 'lep_Eta_1', 'lep_Phi_1', 'Mll01', 'DRll01', 'Ptll01', 'lead_jetPt', 'lead_jetEta', 'lead_jetPhi', 'sublead_jetPt', 'sublead_jetEta', 'sublead_jetPhi', 'HT', 'nJets_OR_T', 'nJets_OR_T_MV2c10_70', 'MET_RefFinal_et', 'MET_RefFinal_phi', 'DRlj00', 'DRjj01']

for inf in inFiles:
    f = rootpy.io.root_open(inf)
    #if 'mc16a' in inf:
    #    mc = 'mc16a'
    #elif 'mc16d' in inf:
    #    mc = 'mc16d'
    #else:
    #    print('no mc16a/d')
    #    break
    #dsid = inf.split('/')[-1]
    #dsid = dsid.replace('.root', '')
    #print(dsid)
    tmpTree = f.get('nominal')
    ROOT.gROOT.cd()
    oldTree = tmpTree.CopyTree('is2LSS0Tau==1')
    oldTree.SetBranchStatus("*",0)
    for br in branch_list:
        oldTree.SetBranchStatus(br,1)

    newFile = rootpy.io.root_open('small.root', 'recreate')
    newTree = oldTree.CloneTree(0)
    for br in branch_list:
        newTree.GetBranch(br).SetFile('small.root')
        newTree.CopyEntries(oldTree)

    newFile.Write()
    newFile.Close()

    g = rootpy.io.root_open('small.root')
    gTree = g.get('nominal')

    gTree.csv(stream=open('87_GN2_new.csv', 'w'))
