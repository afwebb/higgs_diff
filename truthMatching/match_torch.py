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

inFile = sys.argv[1]
#outDir = sys.argv[2]
#njet = sys.argv[2]


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

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

X = normalize(X)
Y = normalize(Y)

X_test = normalize(X_test)
Y_test = normalize(Y_test)

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
#opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion = nn.CrossEntropyLoss()

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
        self.out = nn.Linear(nodes, 2)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        h1 = self.dout(self.relu1(self.fc1(input_)))
        for i in range(self.layers):
            h1 = self.dout(self.relu1(self.fc(h1)))
        a1 = self.out(h1)
        y = self.out_act(a1)
        return y

def train_epoch_batch(model, opt, criterion, batch_size=5000):
    model.train()
    #losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = Y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        # (2) Compute diff
        #print(y_hat[:,0],y_hat[:,1])
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        #losses.append(loss.data.numpy())
    
    y_hat = net(X)
    loss = criterion(y_hat, Y)
    return loss, y_hat

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
    

num_epochs = [200]
nLayers = [4]#, 6, 9]
nNodes = [50]#, 75, 125]#, 100, 175, 250]#[250, 350, 450, 600]

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
    criterion = nn.CrossEntropyLoss()
    y_pred = []
    y_test_pred=[]
    e_losses = []
    test_losses = []
    
    for e in range(p.epochs):
        e_loss, y_pred = train_epoch_batch(net, opt, criterion) 
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
    p.y_pred = y_pred.float().detach().numpy()
    #p.auc = sk.metrics.roc_auc_score(y_train,y_predicted)
    
    print("Nodes: "+str(p.nodes))
    print("Layers: "+str(p.layers))
    print("Train Loss: "+str(p.train_loss))
    print("Test Loss: "+str(p.test_loss))
    print("")

    torch.save(net.state_dict(), 'models/model_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred_test, 'models/y_test_pred_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred, 'models/y_train_pred_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    
    del net, opt, criterion, y_pred, y_pred_test

    plt.figure()
    plt.plot(e_losses, label='train loss')
    plt.plot(test_losses, label='test_loss')
    plt.title("pyTorch Loss, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.test_loss))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend(loc='upper right')
    plt.savefig('plots/torch_loss_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    plt.figure()
    #for c in cutoff:
    yTrain = np.where(Y > 0.5, 1, 0)
    ypTrain = p.y_pred
    
    auc = sk.metrics.roc_auc_score(yTrain,ypTrain[:,1])
    fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain[:,1])
    
    plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
    
    plt.title("pyTorch Train ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
    plt.legend(loc='lower right')    
    plt.savefig('plots/torch_train_roc_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    plt.figure()
    yTest = np.where(Y_test > 0.5, 1, 0)
    ypTest = p.y_pred_test
    
    auc = sk.metrics.roc_auc_score(yTest,ypTest[:,1])
    fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest[:,1])
    
    plt.plot(fpr, tpr, label='AUC = %.3f' %(auc))
    
    plt.title("pyTorch Test ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
    plt.legend(loc='lower right')    
    plt.savefig('plots/torch_test_roc_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    y_test_bin = np.where(ypTest > 0.5, 1, 0)
    print(y_test_bin)
    print('Confusion Matrix:', sklearn.metrics.confusion_matrix(yTest, y_test_bin[:,1]))
