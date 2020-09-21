'''
Given an input variable name, return the name to appear in plots with correct units, range of the variable
Need to convert variables in MeV to GeV for them to display properly
Usage: <display name> <range> = name(<variable>) 
'''

def name(c):
    
    #Give the correct range for the input variable
    if c =='match':                                                                                            
        r =(0,1)#continue                                                                                                  
    elif 'MV2c10' in c or 'DL1r' in c:
        r = (0, 5) 
    elif 'ID' in c:                                                                                        
        r = (-16, 16)      
    elif 'Phi' in c or 'phi' in c:                                                                             
        r = (-3.5, 3.5)                                                                               
    elif 'Eta' in c or 'eta' in c:                                                                                        
        r = (-3.5, 3.5)
    elif 'dR' in c or 'DR' in c:                               
        r = (0,6)        
    elif "score" in c or "Score" in c:
        r = (0,1)                                                                                                        
    elif 'nJets' in c:                                                                                                   
        r = (0,12)
    elif 'nJets' in c and 'DL1r' in c:
        r = (0,5)
    elif c=='HT': 
        r = (0,1500)
    elif c=='HT_jets':
        r = (0,800)
    elif c[0]=='M' and c[-3:]=='met':
        r = (0, 1200)
    elif 'type' in c:
        r = (0, 5)                                                                                           
    elif 'numTrk' in c:
        r = (0, 20)
    elif 'charge' in c:
        r = (-1, 1)
    else:                                                                                               
        r = (0, 400)
    
    #Transform the input variable name to a display name with correct units - generally GeV
    c = c.replace('_', ' ')
    c = c.replace('l0','l_0').replace('l1','l_1').replace('l2','l_2')
    c = c.replace('j0','j_0').replace('j1','j_1')
    c = c.replace('t0','t_0').replace('t1','t_1')
    c = c.replace('lT0','l_{T0}').replace('lT1','l_{T1}')
    c = c.replace('lH0','l_{H0}').replace('lH1','l_{H1}')
    c = c.replace('lT','l_{T}').replace('lH','l_{H}').replace('lW', 'l_W')

    #if c == 'scale_nom' or c == 'match' or c == 'decay': continue
    if 'MhiggsCand' in c:
        c = 'M(ljj) [GeV]'
    if 'Pt ' in c:
        c = c.replace('Pt ', ' $p_T$ ')
        c = c + ' [GeV]'
    if 'pt' in c or 'Pt' in c:
        c = c.replace('pt', ' $p_T(')
        c = c.replace('Pt', ' $p_T(')
        c = c + ')$ [GeV]'
    if 'lep' in c:
        c = c.replace("lep", "Lepton")
    if 'DR' in c:
        c = c.replace(' ', ')(')
        c = c.replace('DR', '$\Delta R(').replace('()', '')
        c = c + ')$'
    if 'dR' in c:
        c = c.replace(' ', ')(')
        c = c.replace('dR', '$\Delta R(').replace('()', '')
        c = c + ')$'
    if 'Eta' in c:
        c = c.replace('Eta', ' $\eta$')
    if 'dPhi' in c:
        c = c.replace('dPhi', '$\Delta\phi(')
        c = c.replace(' ', ')(').replace('()', '') + ')$'
    if 'Ml' in c or 'Mt' in c or 'Mj' in c:
        c = c.replace('M', '$M(')
        c = c + ')$ [GeV]'
    if 'HT' in c:
        c = c + ' [GeV]'
    if c == 'MET RefFinal et' or c=='met':
        c = '$E_T^{miss}$ [GeV]'
    if c == 'MET RefFinal phi' or c=='met phi':
        c = '$E_T^{miss}$ $\phi$'
    if c == 'nJets OR T' or c == 'nJets OR':
        c = 'nJets'
    c = c.replace('met', 'E_T^{miss}')
    #if 'nJets DL1r' in c:
    #    c = 'n b-jets'

    return c, r
