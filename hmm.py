#encoding=utf-8
__author__ = "Jie"
import numpy as np
from sklearn import hmm
import sys
import traceback
import os

def load_single_data (file):
    return [[float(x) for x in l.strip()[1:-1].split(';')] for l in open(file)]

def train(X):
    #print X
    startprob = np.array([0.6, 0.3, 0.1])
    transmat = np.array([[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.3, 0.3, 0.4]])
    model = hmm.GaussianHMM(n_components=3,
                            covariance_type="diag",
                            startprob=startprob,
                            transmat=transmat,
                            algorithm="viterbi",
                            n_iter=100)
    try:
        model.fit(X)
    except:
        traceback.print_exc(file=sys.stder)
    return model 

def test(model, x):
    hs = model.predict(x)
    transmat = model.transmat_
    p = 1
    print "hidden states",hs
    for i in range(len(hs)-1):
        p*=transmat[hs[i]][hs[i+1]]
    print "probability",p

if __name__ == "__main__":
    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    train_data_file = filter(lambda x:x[0]!='.', os.listdir(train_dir))
    test_data_file = filter(lambda x:x[0]!='.', os.listdir(test_dir))
    TR = [np.array(load_single_data(train_dir+'/'+file)) for file in train_data_file]
    TE = [np.array(load_single_data(test_dir+'/'+file)) for file in test_data_file]
    model = train(TR)
    for i in range(len(TE)):
        print test_data_file[i]
        test(model,TE[i])
    




