import pandas as pd
import numpy as np
import re
import sklearn
import h2o
h2o.init()
from h2o.estimators.deeplearning import H2ODeepWaterEstimator
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

#convert to tensors
#h2o_test = h2o.H2OFrame(test)
#h2o_test[3] = h2o_test[3].asfactor()

#h2o_train = h2o.H2OFrame(train)
#h2o_train[3] = h2o_train[3].asfactor()

layers = [4, 6, 9, 12]
nodes = [150, 250, 350]
hidden_grid = []

for i in layers:
    for j in nodes:
        hidden_grid.append( np.full(i, j).tolist() )

param_grid = {
    'hidden':hidden_grid,
    'epochs':[500],
    'train_samples_per_iteration':[500],
    'rate':[0.1],
    'l1':[1e-5],
    'l2':[1e-5],
    'max_w2':[10],
    #'stopping_tolerance':[1e-4],
    #'loss':'mse'
}

nn_grid = H2OGridSearch(
    model = H2ODeepWaterEstimator,
    hyper_params=param_grid,
    grid_id = 'nn_result_4j_87' #str(njet)+'j_'+str(dsid),
)

#nn_grid#.set_params(hidden=[150,150,150,150,150,150,150,150], epochs=1000, train_samples_per_iteration=10000)
nn_grid.train(x=h2o_train.names[1:], y = h2o_train.names[3], 
              training_frame = h2o_train, 
              validation_frame = h2o_test)

print(nn_grid.get_grid(sort_by='mse', decreasing=True))
best_model = nn_grid.models[0]

#model_path = h2o.save_model(nn_grid, path = 'nn_grid_results', force=True)

for mod in nn_grid.models:
    model_path = h2o.save_model(mod, path = 'nn_grid_results/'+njet+'j_'+dsid+'/', force=True)
    print(model_path)

plt.figure()
plt.rcdefaults()
fig, ax = plt.subplots()
variables = best_model._model_json['output']['variable_importances']['variable']
y_pos = np.arange(len(variables))
scaled_importance = best_model._model_json['output']['variable_importances']['scaled_importance']
ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(variables)
ax.invert_yaxis()
ax.set_xlabel('Scaled Importance')
ax.set_title('Variable Importance')
plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+njet+'_'+dsid+'/h2o_var_important.png')

#plt.figure()
#perf = best_model.model_performance()
#perf.plot()
#plt.savefig('/data_ceph/afwebb/higgs_diff/ml_scripts/plots/'+njet+'_'+dsid+'/h2o_roc.png')
