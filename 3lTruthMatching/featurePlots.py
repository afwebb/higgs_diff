import pandas
import xgboost as xgb
import pickle
import pandas as pd
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from namePlot import name
#get_ipython().run_line_magic('matplotlib', 'inline')

inFile = sys.argv[1]
outDir = sys.argv[2]

inDF = pd.read_csv(inFile)#, nrows=100000)
inDF = inDF.dropna()

good = inDF[inDF['match']==1]
bad = inDF[inDF['match']==0]

print(good.shape)
print(bad.shape)

texfile = open('plots/feature_'+outDir+"_plots.tex", "w")

print('\\documentclass[hyperref={pdfpagelayout=SinglePage}]{beamer}\\usetheme{Warsaw}\\usepackage{euler}\\usepackage{pgf}\\usecolortheme{crane}\\usefonttheme{serif}\\useoutertheme{infolines}\\usepackage{epstopdf}\\usepackage{xcolor}\\usepackage{multicol}\\title{Plots}', file=texfile)
print('\\begin{document}', file=texfile)

icount = 1

for c in inDF:
    if c =='match': 
        continue
    elif 'MV2c10' in c:
        r = (-1, 1)
    elif 'ID' in c:
        r = (-16, 16)
    elif 'Phi' in c or 'phi' in c:
        r = (-3.5, 3.5)
    elif 'Eta' in c:
        r = (-3.5, 3.5)
    elif 'dR' in c:
        r = (0, 6)
    elif 'DR' in c:
        r = (0,6)
    elif "score" in c:
        r = (0,1)
    #elif "score" in c:
    #    r = (0,1)
    elif 'nJets' in c:
        r = (0,12)
    elif c=='HT':
        r = (0,1800000)
    elif 'type' in c:
        r = (0, 5)
    elif 'numTrk' in c:
        r = (0, 20)
    elif 'Score' in c:
        r = (0, 1)
    else:
        r = (0, 400000)

    xName = name(c)

    plt.figure()
    plt.hist(good[c], 30, alpha=0.5, range=r, label="Correct")
    plt.hist(bad[c][:good.shape[0]], 30 ,range = r, alpha=0.5, label="Wrong")
    #plt.hist(good[c], 30, alpha=0.5, label="Signal")                                                                                   
    #plt.hist(bad[c][:good.shape[0]], 30, alpha=0.5, label="Background")
    plt.legend()
    plt.xlabel(xName)
    plt.ylabel('NEvts')
    plt.savefig('plots/'+outDir+'/'+c+".pdf")
    plt.close()

    if icount % 4 == 1:
        print ('\\frame{\\frametitle{Validation Plots - '+outDir+'}\n', file=texfile)

    print (r'\includegraphics[width=.42\linewidth]{%s}' % (outDir+'/'+c+".pdf") + ('%'if (icount % 2 == 1) else r'\\'), file=texfile)

    if icount % 4 == 0:
        print ('}\n', file=texfile)
        
    icount += 1

if icount %4 != 1:
    print ('}\n', file=texfile)

f = plt.figure(figsize=(19,15))
plt.matshow(inDF.corr(),fignum=f.number)
for (i,j), z in np.ndenumerate(inDF.corr()):
    plt.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
plt.xticks(range(inDF.shape[1]), inDF.columns, fontsize=14, rotation=45)
plt.yticks(range(inDF.shape[1]), inDF.columns, fontsize=14)
cb = plt.colorbar()
#cb.ax.tick
#plt.title('Top Match Feature Correlations', fontsize=16)
plt.savefig('plots/'+outDir+'/CorrMat.pdf')
#plt.savefig(outDir+"CorrMat.png")
#plt.close()

print ('\\frame{\\frametitle{Validation Plots - '+outDir+'}\n', file=texfile)
print (r'\includegraphics[width=.92\linewidth]{%s}' % (outDir+'/CorrMat.pdf'), file=texfile)
print ('}\n', file=texfile)

print ('\end{document}', file=texfile)
texfile.close()
