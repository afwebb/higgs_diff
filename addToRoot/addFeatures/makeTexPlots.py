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

inDF = pd.read_csv(inFile, nrows=1e3)
inDF = inDF.dropna()

texfile = open('texPlots_'+outDir+".tex", "w")

print('\\documentclass[hyperref={pdfpagelayout=SinglePage}]{beamer}\\usetheme{Warsaw}\\usepackage{euler}\\usepack\age{pgf}\\usecolortheme{crane}\\usefonttheme{serif}\\useoutertheme{infolines}\\usepackage{epstopdf}\\usepackage{xcolor}\\usep\ackage{multicol}\\title{Plots}', file=texfile)
print('\\begin{document}', file=texfile)

icount = 1
for c in inDF:
    if icount % 6 == 1:
        print('\\frame{', file=texfile)
    print(r'\includegraphics[width=.47\linewidth]{%s}' % ('plots/'+outDir+'/features/'+c+".pdf") + ('%'if (icount % 3 != 0) else r'\\'), file=texfile)
    if icount % 6 == 0:
        print('}\n', file=texfile)
    icount += 1

if icount %4 != 1:
    print('}\n', file=texfile)

print('\end{document}', file=texfile)
texfile.close()
