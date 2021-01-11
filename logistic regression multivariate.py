import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv('iris.csv',
                    names=['index,length of sepal','width of sepal','length of patal','width of patal','type'])


x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.preprocessing import *
lb=LabelEncoder()
y=lb.fit_transform(y)

from sklearn.model_selection import *

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.5,random_state=0)

n_class=len(np.unique(y))
theta=np.random.rand(train_x.shape[1]+1,n_class)

def softmax(theta,x):
    res=x.dot(theta)
    return np.exp(res)/np.sum(np.exp(res),axis=1,keepdims=True)

def y_train_one_hot(y):
    m=len(y)
    first=np.zeros((m,n_class))
    
    first[np.arange(m),y]=1
    return first


def softmax_func(theta,x,y):
    it=500
    lr=0.1
    ep=1e-7
    n=len(x)
    for i in range(it):
        y_prob=softmax(theta,x)
        one_hot=y_train_one_hot(y)
        loss=np.mean(np.sum(one_hot*np.log(y_prob+ep),axis=1))
        if i%500==0:
            print(i,':',loss)
        error=y_prob-one_hot
        gradients=1/n*x.T.dot(error)
        theta=theta-lr*gradients
    return theta

x_t=np.append(np.ones((train_x.shape[0],1)),train_x,axis=1)

res=softmax_func(theta,x_t,train_y)

logits=softmax(res,x_t)
y_pred=np.argmax(logits,axis=1)

acc=np.mean(train_y==y_pred)















