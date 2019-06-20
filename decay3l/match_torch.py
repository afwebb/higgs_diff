import pandas as pd
import numpy as np
import sklearn
import sklearn as sk
from sklearn.model_selection import train_test_split
import torch
import math
import sys
import os
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import scipy

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

device = torch.device("cuda:0")

#inFile = sys.argv[1]
outDir = sys.argv[1]
#njet = sys.argv[2]

'''
inDF = pd.read_csv(inFile)

inDF = sk.utils.shuffle(inDF)

inDF[abs(inDF) < 0.001] = 0

pd_train, pd_test = train_test_split(inDF, test_size=0.3)

y_test = pd_test['match']
y_train = pd_train['match']
pd_train = pd_train.drop(['match'],axis=1)                                                                                                         
pd_test = pd_test.drop(['match'],axis=1) 

#Convert data to tensors
x_train = torch.tensor(pd_train.values, dtype=torch.float32)
x_test = torch.tensor(pd_test.values, dtype=torch.float32)
y_train = torch.FloatTensor(y_train.values).long()
y_test = torch.FloatTensor(y_test.values).long()

X = x_train
Y = y_train

X_test = x_test
Y_test = y_test
'''

X = torch.load('tensors/torch_x_train_'+outDir+'.pt')
X_test = torch.load('tensors/torch_x_test_'+outDir+'.pt')

Y = torch.load('tensors/torch_y_train_'+outDir+'.pt')
Y_test = torch.load('tensors/torch_y_test_'+outDir+'.pt')

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

xMax = X.max(0, keepdim=True)[0]
yMax = Y.max(0, keepdim=True)[0]

X = X / xMax
Y = Y / yMax #normalize(Y)

X_test = X_test / xMax #normalize(X_test)
Y_test = Y_test / yMax #normalize(Y_test)

print('past normal')

class OldNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y
    
oldNet = OldNet()
#opt = optim.Adam(net.parameters(), lr=0.002, betas=(0.9, 0.999))
criterion = nn.BCELoss()#nn.CrossEntropyLoss()

class Net(nn.Module):
    
    def __init__(self, D_in, nodes, layers):
        self.layers = layers
        super().__init__()
        self.fc1 = nn.Linear(D_in, nodes)
        self.relu1 = nn.LeakyReLU()
        self.dout = nn.Dropout(0.2)
        #self.fc2 = nn.Linear(50, 100)                                                                                                           
        self.fc = nn.Linear(nodes, nodes)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(nodes, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        h1 = self.dout(self.relu1(self.fc1(input_)))
        for i in range(self.layers):
            h1 = self.dout(self.relu1(self.fc(h1)))
        a1 = self.out(h1)
        y = self.out_act(a1)
        return y

def train_epoch_batch(model, opt, criterion, batch_size=10000):
    model.train()
    #losses = []
    #y_hat = torch.empty(1, 2)
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat_batch = net(x_batch)

        #y_hat = torch.cat((y_hat, y_hat_batch), 0)
        # (2) Compute diff
        #print(y_hat[:,0],y_hat[:,1])
        loss = criterion(y_hat_batch, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        #losses.append(loss.data.numpy())
    
    #y_hat = net(X)
    #loss = criterion(y_hat, Y)
    return loss

def train_epoch(model, opt, criterion, batch_size=5000):
    model.train()
    #losses = []
    opt.zero_grad()
    # (1) Forward
    y_hat = net(X)
    # (2) Compute diff
    loss = criterion(y_hat, Y)
    # (3) Compute gradients
    loss.backward()
    # (4) update weights
    opt.step()        
    #losses.append(loss.data.numpy())
    
    return loss, y_hat

class param:
    def __init__(self, epochs, layers, nodes, auc = 0, loss = 1):
        self.epochs = epochs
        self.layers = layers
        self.nodes = nodes
        self.auc = auc
        self.train_loss = None
        self.test_loss = None
        self.y_pred = None
        self.y_pred_test = None
        self.net = None
    

num_epochs = [150]
nLayers = [8]#, 5, 6]#, 6, 9]
nNodes = [100]#, 50, 75, 100]#, 75, 125]#, 100, 175, 250]#[250, 350, 450, 600]

param_grid = []
for ep in num_epochs:
    for la in nLayers:
        for node in nNodes:
            param_grid.append(param(ep, la, node))

def scale_pt(y_predicted):
    y_pred_scaled = (y_predicted-y_predicted.min())/(y_predicted.max()-y_predicted.min())
    return y_pred_scaled

for p in param_grid:
    net = Net(X.size()[1], p.nodes, p.layers)
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    #opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.BCELoss()#nn.CrossEntropyLoss()
    y_pred = []
    y_test_pred=[]
    e_losses = []
    test_losses = []
    
    for e in range(p.epochs):
        e_loss = train_epoch_batch(net, opt, criterion) 
        #e_losses.append(loss)
        if e%5==0:
            y_pred_test = net(X_test)
            test_loss = criterion(y_pred_test, Y_test).float().detach().numpy()
            test_losses.append(test_loss)
            e_losses.append(e_loss.float().detach().numpy())
            print("[Epoch]: %i, [Train Loss]: %.4f, [Test Loss]: %.4f" % (e, e_loss, test_loss))
            #if e>3 and test_losses[-2]-test_losses[-1]<10e-8 and test_losses[-3]-test_losses[-1]<10e-8:
            #    p.epochs=e
            #    break
    
    p.net = net
    p.train_loss = e_loss.float().detach().numpy()
    p.test_loss = test_loss
    p.y_pred_test = y_pred_test.float().detach().numpy()
    print('past pred_test')
    p.y_pred = []
    batch_size=50000
    for beg_i in range(0, X.size(0), batch_size):
        tmp_pred = net(X[beg_i:beg_i + batch_size, :])[:,0].float().detach().numpy()
        p.y_pred = np.concatenate((p.y_pred, tmp_pred), axis=0)

    #p.y_pred = net(X).float().detach().numpy()
    print('past net')
    #p.auc = sk.metrics.roc_auc_score(y_train,y_predicted)
    
    print('test pred: ', y_pred_test)

    print("Nodes: "+str(p.nodes))
    print("Layers: "+str(p.layers))
    print("Train Loss: "+str(p.train_loss))
    print("Test Loss: "+str(p.test_loss))
    print("")

    torch.save(p.net.state_dict(), 'models/model_'+outDir+'_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred_test, 'models/y_test_pred_'+outDir+'_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred, 'models/y_train_pred_'+outDir+'_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    
    del net, opt, criterion, y_pred, y_pred_test

    plt.figure()
    plt.plot(e_losses, label='train loss')
    plt.plot(test_losses, label='test_loss')
    plt.title("pyTorch Loss, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.test_loss))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.savefig('plots/torch_'+outDir+'_loss_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    plt.figure()
    #for c in cutoff:
    yTrain = np.where(Y > 0.5, 1, 0)
    ypTrain = p.y_pred
    
    auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
    fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)
    
    plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
    
    plt.title("pyTorch Train ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
    plt.legend(loc='lower right')    
    plt.savefig('plots/torch_'+outDir+'_train_roc_'+str(p.layers)+'l_'+str(p.nodes)+'.png')

    plt.figure()
    yTest = np.where(Y_test > 0.5, 1, 0)
    ypTest = p.y_pred_test
    
    auc = sk.metrics.roc_auc_score(yTest,ypTest)
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)
    
    plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
    
    plt.title("pyTorch Test ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
    plt.legend(loc='lower right')    
    plt.savefig('plots/torch_'+outDir+'_test_roc_'+str(p.layers)+'l_'+str(p.nodes)+'.png')

    testPredTrue = p.y_pred_test[yTest==1]
    testPredFalse = p.y_pred_test[yTest==0]
    
    trainPredTrue = p.y_pred[yTrain==1]
    trainPredFalse = p.y_pred[yTrain==0]

    plt.figure()
    plt.hist(testPredTrue, 30, log=False, alpha=0.5, label='Semilep')
    plt.hist(testPredFalse[:len(testPredTrue)], 30, log=False, alpha=0.5, label='Full lep')
    plt.title("DNN Output, Test Data")
    plt.xlabel('DNN Score')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/torch_'+outDir+'_'+str(p.layers)+'l_'+str(p.nodes)+'_test_score.png')
    
    plt.figure()
    plt.hist(trainPredTrue, 30, log=False, alpha=0.5, label='Semilep')
    plt.hist(trainPredFalse[:len(trainPredTrue)], 30, log=False, alpha=0.5, label='Full lep')
    plt.title("DNN Output, Train Data")
    plt.xlabel('DNN Score')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/torch_'+outDir+'_'+str(p.layers)+'l_'+str(p.nodes)+'_train_score.png')
    
    y_test_bin = np.where(ypTest > 0.5, 1, 0)
    print(y_test_bin)
    print('Confusion Matrix:', sklearn.metrics.confusion_matrix(yTest, y_test_bin))
