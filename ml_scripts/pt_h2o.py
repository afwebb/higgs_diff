import pandas as pd
import numpy as np
import re
import sklearn
import h2o
h2o.init()
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.grid import H2OGridSearch
#import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import sys
import os

h2o.remove_all()

dsid = sys.argv[1]
njet = sys.argv[2]
'''
#Reading the data
#inDF = pd.read_csv('../inputData/67_4j.csv')
inDF = pd.read_csv('../inputData/'+dsid+'_'+njet+'j.csv')

#change higgs_pt from continuous to categorical
#inDF.loc[ inDF['higgs_pt'] <= 300000, 'higgs_pt'] = 0
#inDF.loc[ inDF['higgs_pt'] > 300000, 'higgs_pt'] = 1 ;

inDF.shape

train, test = train_test_split(inDF, test_size=0.3)
train.columns
train.shape
test.shape

#train = train.drop(['higgs_pt'],axis=1)
#test = test.drop(['higgs_pt'],axis=1)
'''

h2o_train = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+dsid+'_'+njet+'.csv', destination_frame='h2o_train')
h2o_test = h2o.import_file(path='/data_ceph/afwebb/higgs_diff/inputData/tensors/h2o_train_'+dsid+'_'+njet+'.csv', destination_frame='h2o_test')

y_train = h2o_train.names[3]
y_test = h2o_test.names[3]

layers = [4, 6, 9, 12]
nodes = [150, 250, 350, 450]
hidden_grid = []

for i in layers:
    for j in nodes:
        hidden_grid.append( np.full(i, j).tolist() )

param_grid = {
    'hidden':[50,50]#[450,450,450,450,450,450]
    #'epochs': 2000,
    #'train_samples_per_iteration': 1000,
    #'rate': 0.01,
    #'l1':1e-5,
    #'l2':1e-5,
    #'max_w2':10,
    #'stopping_tolerance':1e-5,
    #'loss':'mse'                                                                                                                                
}

h2o_model = H2ODeepLearningEstimator(hidden=[20,20],#[450,450,450,450,450,450]
                                     epochs= 20,
                                     train_samples_per_iteration= 1000,
                                     rate= 0.01,
                                     #l1=1e-5,
                                     #l2=1e-5,
                                     #max_w2=10,
                                     stopping_tolerance=1e-5)
                                     

#nn_grid#.set_params(hidden=[150,150,150,150,150,150,150,150], epochs=1000, train_samples_per_iteration=10000)
h2o_model.train(x=h2o_train.names[1:], y = h2o_train.names[3], 
                training_frame = h2o_train, 
                validation_frame = h2o_test)

print(h2o_model.mse(valid=True))

model_path = h2o.save_model(h2o_model, path = 'h2o_models/new_4j_87/', force=True)

plt.figure()
plt.rcdefaults()
fig, ax = plt.subplots()
variables = h2o_model._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = h2o_model._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+njet+'_'+dsid+'/h2o_var_important.png')

y_train_pred = h2o_model.predict(h2o_train)
y_test_pred = h2o_model.predict(h2o_test)

y_train = h2o_train[3].as_data_frame().as_matrix()
y_test = h2o_test[3].as_data_frame().as_matrix()

y_train_pred = y_train_pred.as_data_frame().as_matrix()
y_test_pred = y_test_pred.as_data_frame().as_matrix()

plt.figure()
plt.hist(y_test, 20, log=False, range=(0,0.8), alpha=0.5, label='truth')
plt.hist(y_test_pred, 20, log=False, range=(0,0.8), alpha=0.5, label='test')
plt.title("H2O Test Data, layers=%i, nodes=%i, loss=%0.4f" %(6, 450, h2o_model.mse(valid=True)))
plt.xlabel('Higgs Pt')
plt.ylabel('NEvents')
plt.legend(loc='upper right')
plt.savefig('plots/'+njet+'_'+dsid+'/h2o_test_pt_spectrum_'+str(6)+'l_'+str(450)+'n.png')

xy = np.vstack([y_test, y_test_pred])
z = scipy.stats.gaussian_kde(xy)(xy)

plt.figure()
plt.scatter(y_test, y_test_pred, c=z, edgecolor='')
plt.title("H2O Test Data, layers=%i, nodes=%i, loss=%0.4f" %(6, 450, h2o_model.mse(valid=True)))
plt.xlabel('Truth Pt')
plt.ylabel('Predicted Pt')
plt.plot([0,0.6],[0,0.6], zorder=10)
plt.savefig('plots/'+njet+'_'+dsid+'/h2o_test_pt_scatter_'+str(6)+'l_'+str(450)+'n.png')

cutoff = [150000, 200000, 250000]

plt.figure()
for c in cutoff:
    yTest = np.where(y_test > c, 1, 0)
    ypTest = y_test_pred

    auc = sk.metrics.roc_auc_score(yTest,ypTest)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

    plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

plt.title("H2O Test ROC")
plt.legend(loc='lower right')
plt.savefig('plots/'+njet+'_'+dsid+'/h2o_test_roc_'+str(6)+'l_'+str(450)+'n.png')

#plt.figure()
#perf = best_model.model_performance()
#perf.plot()
#plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+njet+'_'+dsid+'/h2o_roc.png')
