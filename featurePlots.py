'''
Plot input features of a dataset, correlation matrix
Save as .pdf, and write latex file with plots included in plots/<out string> directory
Usage: python3.6 featurePlots.py <input dataset.csv> <out string>
'''

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

inDF = pd.read_csv(inFile, nrows=1e5)
inDF = inDF.dropna()

good = inDF[inDF['match']==1]
bad = inDF[inDF['match']==0]

print(good.shape)
print(bad.shape)

texfile = open('plots/'+outDir+'/features_'+outDir+".tex", "w")

print('\\documentclass[hyperref={pdfpagelayout=SinglePage}]{beamer}\\usetheme{Warsaw}\\usepackage{euler}\\usepackage{pgf}\\usecolortheme{crane}\\usefonttheme{serif}\\useoutertheme{infolines}\\usepackage{epstopdf}\\usepackage{xcolor}\\usepackage{multicol}\\title{Plots}', file=texfile)
print('\\begin{document}', file=texfile)

icount = 1

for c in inDF:

    xName, r = name(c)
    print(xName)

    if 'GeV' in xName: # Convert MeV to GeV
        good[c] = good[c]/1000
        bad[c] = bad[c]/1000

    nEnt = min(len(good[c]),len(bad[c])) #Plot an equal number of signal and background events

    plt.figure()
    plt.hist(good[c][:nEnt], 30, alpha=0.5, range=r, label="Signal")
    plt.hist(bad[c][:nEnt], 30 ,range = r, alpha=0.5, label="Background")
    plt.legend()
    plt.xlabel(xName)
    plt.ylabel('NEvts')
    plt.savefig('plots/'+outDir+'/features/'+c+".pdf")
    plt.close()

    if icount % 4 == 1:
        print ('\\frame{\\frametitle{Validation Plots - '+outDir+'}\n', file=texfile)

    print (r'\includegraphics[width=.42\linewidth]{%s}' % ('features/'+c+".pdf") + ('%'if (icount % 2 == 1) else r'\\'), file=texfile)

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
plt.savefig('plots/'+outDir+'/features/CorrMat.pdf')
#plt.savefig(outDir+"CorrMat.png")
#plt.close()

print ('\\frame{\\frametitle{Validation Plots - '+outDir+'}\n', file=texfile)
print (r'\includegraphics[width=.92\linewidth]{%s}' % ('features/CorrMat.pdf'), file=texfile)
print ('}\n', file=texfile)

print ('\end{document}', file=texfile)
texfile.close()
