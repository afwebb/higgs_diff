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

outDir = sys.argv[1]
#njet = sys.argv[2]

X = torch.load('../inputData/tensors/torch_x_train_'+outDir+'.pt')
X_test = torch.load('../inputData/tensors/torch_x_test_'+outDir+'.pt')

Y = torch.load('../inputData/tensors/torch_y_train_'+outDir+'.pt')
Y_test = torch.load('../inputData/tensors/torch_y_test_'+outDir+'.pt')

yMax = Y.max(0, keepdim=True)[0]
xMax = X.max(0, keepdim=True)[0]

def normalize(x):
    x_normed = x / x.max(0, keepdim=True)[0]
    return x_normed

X = X / xMax #normalize(X)
Y = Y / yMax #normalize(Y)

X_test = X_test / xMax#X.max(0, keepdim=True)[0]#normalize(X_test)
Y_test = Y_test / yMax#normalize(Y_test)

normFactors = xMax.float().detach()#.numpy() #[*yMax.float().detach().numpy(), *xMax.float().detach().numpy()]
normFactors = np.insert(normFactors, 0, yMax.float().detach())#.numpy())
print(normFactors)
normFactors = np.asarray(normFactors)
np.save('torch_models/'+outDir+'/normFactors.npy', normFactors)

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
#criterion = nn.BCELoss()

class Net(nn.Module):
    
    def __init__(self, D_in, nodes, layers):
        self.layers = layers
        super().__init__()
        self.fc1 = nn.Linear(D_in, nodes)
        self.relu1 = nn.LeakyReLU()
        self.dout = nn.Dropout(0.25)
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

def train_epoch_batch(model, opt, criterion, batch_size=2000):
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
        loss = criterion(y_hat[:,0], y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()        
        #losses.append(loss.data.numpy())
    
    y_hat = net(X)
    loss = criterion(y_hat[:,0], Y)
    return loss, y_hat[:,0]

def train_epoch(model, opt, criterion, batch_size=5000):
    model.train()
    #losses = []
    opt.zero_grad()
    # (1) Forward
    y_hat = net(X)
    # (2) Compute diff
    loss = criterion(y_hat[:,0], Y)
    # (3) Compute gradients
    loss.backward()
    # (4) update weights
    opt.step()        
    #losses.append(loss.data.numpy())
    
    return loss, y_hat[:,0]

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
    

num_epochs = [300]
nLayers = [6]#[2, 3, 4, 6, 8]
nNodes = [75]#[25, 50, 75, 100]#[50, 75, 125]#, 100, 175, 250]#[250, 350, 450, 600]

lnPairs = [[6,100], [7,100]]#[[5,100],[6,75],[6,100],[8,50],[8,75]]

param_grid = []
for ep in num_epochs:
    for x in lnPairs:
        la = x[0]
        node = x[1]
    #for la in nLayers:
        #for node in nNodes:
        param_grid.append(param(ep, la, node))

def scale_pt(y_predicted):
    y_pred_scaled = (y_predicted-y_predicted.min())/(y_predicted.max()-y_predicted.min())
    return y_pred_scaled

for p in param_grid:
    net = Net(X.size()[1], p.nodes, p.layers)
    opt = optim.Adam(net.parameters(), lr=0.0005, betas=(0.9, 0.999))
    #opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.MSELoss()
    y_pred = []
    y_test_pred=[]
    e_losses = []
    test_losses = []
    
    for e in range(p.epochs):
        e_loss, y_pred = train_epoch_batch(net, opt, criterion) 
        #e_losses.append(loss)
        if e%5==0:
            y_pred_test = net(X_test)[:,0]
            test_loss = criterion(y_pred_test, Y_test).float().detach().numpy()
            test_losses.append(test_loss)
            e_losses.append(e_loss.float().detach().numpy())
            print("[Epoch]: %i, [Train Loss]: %.5f, [Test Loss]: %.5f" % (e, e_loss, test_loss))
            #if e>3 and test_losses[-2]-test_losses[-1]<10e-8 and test_losses[-3]-test_losses[-1]<10e-8:
            #    p.epochs=e
            #    break
    
    p.net = net
    p.train_loss = np.sqrt(e_loss.float().detach().numpy())*yMax
    p.test_loss = np.sqrt(test_loss)*yMax
    p.y_pred_test = y_pred_test.float().detach().numpy()*yMax
    p.y_pred = y_pred.float().detach().numpy()*yMax
    #p.auc = sk.metrics.roc_auc_score(y_train,y_predicted)

    Y_full = Y*yMax
    Y_full_test = Y_test*yMax
    
    print("Nodes: "+str(p.nodes))
    print("Layers: "+str(p.layers))
    print("Train Loss: "+str(p.train_loss))
    print("Test Loss: "+str(p.test_loss))
    print("")

    torch.save(net.state_dict(), 'torch_models/'+outDir+'/model_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred_test, 'torch_models/'+outDir+'/y_test_pred_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')
    torch.save(p.y_pred, 'torch_models/'+outDir+'/y_train_pred_'+str(p.layers)+'l_'+str(p.nodes)+'n.pt')

    del net, opt, criterion, y_pred, y_pred_test

    plt.figure()
    plt.plot(e_losses, label='train loss')
    plt.plot(test_losses, label='test_loss')
    plt.title("pyTorch Loss, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.test_loss))
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+outDir+'/torch_loss_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    plt.figure()
    plt.hist(Y_full, 30, log=True, range=(0,800000), alpha=0.5, label='truth')
    plt.hist(p.y_pred, 30, log=True, range=(0,800000), alpha=0.5, label='train')
    plt.title("pyTorch Train Data, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.train_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+outDir+'/torch_train_pt_spectrum_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    plt.figure()
    plt.hist(Y_full_test, 30, log=True, range=(0,800000), alpha=0.5, label='truth')
    plt.hist(p.y_pred_test,30, log=True, range=(0,800000), alpha=0.5, label='test')
    plt.title("pyTorch Test Data, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.test_loss))
    plt.xlabel('Higgs Pt')
    plt.ylabel('NEvents')
    plt.legend(loc='upper right')
    plt.savefig('plots/'+outDir+'/torch_test_pt_spectrum_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    #ROC
    cutoff = [150000]

    plt.figure()
    for c in cutoff:
        yTrain = np.where(Y_full > c, 1, 0)
        ypTrain = p.y_pred

        auc = sk.metrics.roc_auc_score(yTrain,ypTrain)
        fpr, tpr, _ = sk.metrics.roc_curve(yTrain,ypTrain)

        plt.plot(fpr, tpr, label='Train AUC = %.3f' %(auc))

        yTest = np.where(Y_full_test > c, 1, 0)
        ypTest = p.y_pred_test

        auc = sk.metrics.roc_auc_score(yTest,ypTest)
        fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

        plt.plot(fpr, tpr, label='Test AUC = %.3f' %(auc))

        plt.title("pyTorch Train ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
        plt.legend(loc='lower right')
    plt.savefig('plots/'+outDir+'/torch_roc_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')
    '''
    plt.figure()
    for c in cutoff:
        yTest = np.where(Y_full_test > c, 1, 0)
        ypTest = p.y_pred_test

        auc = sk.metrics.roc_auc_score(yTest,ypTest)
        fpr, tpr, _ = sk.metrics.roc_curve(yTest,ypTest)

        plt.plot(fpr, tpr, label='AUC = %.3f, cutoff = %0.2f' %(auc, c))

        plt.title("pyTorch Test ROC, layers=%i, nodes=%i" %(p.layers, p.nodes))
        plt.legend(loc='lower right')
    plt.savefig('plots/'+outDir+'/torch_test_roc_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')
    '''

    # Calculate the point density
    xy = np.vstack([Y_full_test[:100000], p.y_pred_test[:100000]])
    z = scipy.stats.gaussian_kde(xy)(xy)

    plt.figure()
    plt.scatter(Y_full_test[:100000], p.y_pred_test[:100000], c=np.log(z), edgecolor='')
    plt.title("pyTorch Test Data, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.test_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,800000],[0,800000], zorder=10)
    plt.savefig('plots/'+outDir+'/torch_test_pt_scatter_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    # Calculate the point density                                                                                      
    xy_train = np.vstack([Y_full[:100000], p.y_pred[:100000] ])
    z_train = scipy.stats.gaussian_kde(xy_train)(xy_train)

    plt.figure()
    plt.scatter(Y_full[:100000], p.y_pred[:100000], c=np.log(z_train), edgecolor='')
    plt.title("pyTorch Train Data, layers=%i, nodes=%i, loss=%0.4f" %(p.layers, p.nodes, p.train_loss))
    plt.xlabel('Truth Pt')
    plt.ylabel('Predicted Pt')
    plt.plot([0,800000],[0,800000], zorder=10)
    plt.savefig('plots/'+outDir+'/torch_train_pt_scatter_'+str(p.layers)+'l_'+str(p.nodes)+'n.png')

    #y_predicted = y_pred.float().detach().numpy()
    
