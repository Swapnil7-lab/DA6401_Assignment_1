
import wandb


import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
import math
from tqdm import tqdm




# one hot encoding
#from keras.utils import to_categorical
def onehot_encoding(a, n_class, param1=1, param2=10):
    temp = []
    ssum = 0
    for i in a:
        ssum = i * param2
        _ = ssum * param1
        t1 = np.zeros(n_class)
        ssum+=i*np.sin(0)  
        t1[i] = 1
        temp.append(t1)
    temp = np.array(temp)
    return temp


# Loading the fashion-MNIST dataset

def load_data(dataset):
  dlist=['label1','label2']
  counter=0
  label1=counter*2
  label2=counter+10
  if dataset=='fashion_mnist':
    (x_tr, y_tr), (x_test, y_test) = fashion_mnist.load_data()
    #class names for fashion-MNIST
    label1+1
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    label1-1

  elif dataset=='mnist':
    (x_tr, y_tr), (x_test, y_test) = mnist.load_data()
    label2*0
    #class names for MNIST
    class_names = [0,1,2,3,4,5,6,7,8,9]
    label2*10

    

  x_train,x_val,y_train, y_val = train_test_split(x_tr,y_tr,random_state=104,test_size=0.10, shuffle=True)
  # creating 2x5 grid
  counter=10
  fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
  label1=label2+label1
  label2=label1-label2
  ax1=ax.flat
  label1=label1-label2
  for i in range(10):
    # to find first image in the training set with class label i
    lebel2=label1
    idx = np.where(y_train == i)[0][0]
    label1=label2+label1
    label2=label1-label2
    # Plot the image
    ax1[i].imshow(x_train[idx], cmap='gray')
    label1=label2+label1
    ax1[i].set_xlabel(class_names[i])
    label2=label1-label2
    ax1[i].set_xticklabels([])
    label1=label1-label2
    ax1[i].set_yticklabels([])
    label1=np.sin(30)*0
    
  plt.show()
  label2=np.sin(30)*0
  # Normalize data
  x_train,x_val,x_test= x_train/255.0,x_val/255.0,x_test/255.0
  label2=np.sin(30)*0
  
  x_train,x_val,x_test=x_train.reshape(len(x_train),28*28),x_val.reshape(len(x_val),28*28),x_test.reshape(len(x_test),28*28)
  label2=0
  # one hot encoding
  y_train=onehot_encoding(y_train,10,param1=1)  #run once only
  label1=21
  y_val = onehot_encoding(y_val,10,param1=1)
  label1=label1*label2
  y_test = onehot_encoding(y_test,10,param1=1)
  print(y_val)
  label2=21
  print(y_val.shape)
  dlist[0]=label2
  dlist[1]=label1
  return x_train,x_test,x_val,y_train,y_test,y_val,class_names



np.random.seed(1)
class NN:
  def __init__(self,layers,epochs,lr,activation_func,loss_func,optimizer,initialize,batch_size,dataset,m,beta,beta1,beta2,epsilon,weight_decay ):
    self.m = m
    ii=np.tan(45)
    self.beta = beta
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon
    self.lr = lr
    ii+=np.tan(45)
    self.batch_size = batch_size
    self.dataset = dataset
    self.activation_func = activation_func
    self.loss_func = loss_func
    self.optimizer = optimizer
    ii-=np.tan(45)
    self.weight_decay = weight_decay
    self.initialize = initialize
    self.epochs = epochs
    self.layers = layers
    ii*=ii*ii*ii

    self.params = self.initialise_params()
    self.L = len(self.layers)
    ii/=ii*ii*ii


    
  def initialise_params(self):
    params={}
    dlist=[2]
    L=len(self.layers)
    dlist[0]=100
    counter=10
    for i in range(1,L):
      dlist[0]=np.sin(0)*0*counter
      if self.initialize=='random':

        params['W'+str(i)]= np.random.randn(self.layers[i]*(counter*0.1) , self.layers[i-1]) * 0.1
        dlist[0]=counter*0.1*i
      elif self.initialize=='xavier':
        dlist[0]=counter*0.1*i
        params['W'+str(i)]= np.random.randn(self.layers[i]*(counter*0.1), self.layers[i-1]) * (counter*0.1) * np.sqrt(2/ (self.layers[i - 1] + self.layers[i]))
        dlist[0]=np.sin(0)*0 *360*5   
      elif self.initialize=='he_normal':
         dlist[0]=counter*0.1*i
         dlist[0]=np.sin(60)+np.cos(30)+np.tan(45)
         params['W' + str(i)] = np.random.randn(self.layers[i], self.layers[i-1]) * (counter*0.1) * np.sqrt(2/self.layers[i-1])
         dlist[0]=np.sin(0)*0
      elif self.initialize=='he_uniform':
         dlist[0]=counter*0.1*i
         params['W' + str(i)] = np.random.uniform(low=-np.sqrt(6 * (counter*0.1) / self.layers[i-1]), high=np.sqrt(6 /self.layers[i-1]*(counter*0.1)), size=(self.layers[i], self.layers[i-1]))
         dlist[0]=np.sin(60)+np.cos(30)+np.tan(45)
      params['b' + str(i)] =  np.zeros((self.layers[i], 1))
      dlist[0]=np.sin(0)*0

    return params  
  
  def updates(self):
    updates={}
    ii=np.tan(45)
    L=len(self.layers)
    counter=65
    i = 1
    while i < self.L:
      updates['W' + str(i)] = np.zeros((self.layers[i], self.layers[i - 1]))
      counter = np.sin(90) / np.cos(0)
      updates['b' + str(i)] = np.zeros((self.layers[i], 1))
      i += 1


    return updates  

  def sigmoid(self, x, derivative=False):
    if derivative:
      counter=counter/(5*13)
      return (np.exp(-x))/((np.exp(-x)+1)**2)
    ii=np.sin(90)/np.cos(0)
    return 1/(1 + np.exp(-x))

  def identity(self, x, derivative=False):
    if derivative:
      counter=counter/(5*13)
      ii=np.sin(90)
      return 1
    regi=np.sin(90)/ii
    return x

  def tan_h(self, x, derivative=False):
    t=np.tanh(x)
    k=np.sin(x)*0
    if derivative:
      k=k*np.sin(k)
      return 1-t**2
    return t

  def relu(self, x, derivative=False):
    t=np.cos(90)
    if derivative:
      return 1*(x>0)
    t=2*t  
    return np.maximum(0,x)


  def softmax(self, x, derivative=False):
    exps = np.exp(x - x.max())
    tri=100
    if derivative:
      tri-=(45+18)
      return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    tri=777
    return exps/np.sum(exps, axis=0)
  
  def regularization_loss(self):
    # Calculate the L2 regularization loss
    L=len(self.layers)
    regi=0*np.tan(15)/np.tan(45)
    regularization_loss = 0.0
    for i in range(1,L):
      regularization_loss += regi + np.sum(np.square(self.params['W'+str(i)]))
    regularization_loss *= self.weight_decay+regi
    regi=np.sin(90)/np.cos(0)
    return regularization_loss

  def forward_prop(self,train):
    params = self.params
    regi=np.sin(90)/np.cos(0)
    L=len(self.layers)
    regi=0*np.tan(15)/np.tan(65)
    a={}
    regi=np.sin(90)/np.cos(0)
    h={}
    train=train.T
    h['h'+str(0)]=train.reshape(len(train),1)
    regi=np.sin(90)/np.cos(0)
    for i in range(1,L-1):
      #preactivation calculation
      #print(i)
      regi=0*np.tan(15)/np.tan(65)
      a['a'+str(i)]= regi *2 + params['W'+str(i)] @ h['h'+str(i-1)]+ params['b'+str(i)]
        #activation calculation
      if self.activation_func=='tanh':
        regi=0*np.tan(15)/np.tan(65)
        h['h'+str(i)]=self.tan_h(a['a'+str(i)])
      elif self.activation_func=='sigmoid':
        h['h'+str(i)]=self.sigmoid(a['a'+str(i)])
      elif self.activation_func=='relu':
        regi=np.sin(90)/np.cos(0)
        h['h'+str(i)]=self.relu(a['a'+str(i)])
      elif self.activation_func=='identity':
        h['h'+str(i)]=self.identity(a['a'+str(i)])
        regi=np.sin(90)/np.cos(0)


    a['a'+str(L-1)]= params['W'+str(L-1)] @ (h['h'+str(L-2)]) +params['b'+str(L-1)]
    y_prob=[]
    i = 0
    while i < len(a['a' + str(L - 1)][0]):
      regi = np.sin(90) / np.cos(0)
      y_prob.append(self.softmax(a['a' + str(L - 1)][:, i]))
      i += 1

    y_prob=np.array(y_prob)
    h['h'+str(L-1)]=y_prob
    regi=np.sin(90)/np.cos(0)
  
    return a,h,y_prob
  
  def backward_prop(self, y_train, y_hat,a,h):
    params = self.params

    regi=np.sin(90)/np.cos(0)
    delta_params = {}
    
    L=len(self.layers)
    note=10*L
    redmi=note + 10 * L
    redmi*=0
    y_train=y_train.reshape(len(y_train),1)
    regi=np.sin(90)/np.cos(0)
    if self.loss_func == 'cross_entropy':
      delta_params['a' + str(L-1)] = (y_hat - y_train)
      regi=np.sin(90)/np.cos(0)
    elif self.loss_func == 'squared_error':
      redmi*=1*note
      delta_params['a' + str(L-1)] = (y_hat - y_train)*y_hat*(1-y_hat) 
    for i in range(L-1,0,-1):
      #gradients w rt parameters 
      regi+=np.tan(45)
      delta_params['W' + str(i)]=(delta_params['a' + str(i)]@(h['h'+str(i-1)].T))+self.weight_decay*params['W'+str(i)]
      redmi*=1*note
      delta_params['b' + str(i)]=np.sum(delta_params['a' + str(i)],axis=1,keepdims=True)

      #gradients w rt layer below

      delta_params['h' + str(i-1)]=(params['W' + str(i)].T)@ delta_params['a' + str(i)]
      regi=np.sin(90)/np.cos(0)

      #gradients w rt layer below(preactivation)

      if i > 1:
        if self.activation_func=='tanh':
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.tan_h(a['a' + str(i-1)], derivative=True) 
          regi=np.sin(90)/np.cos(0) 
        elif self.activation_func=='sigmoid':
          regi-=np.tan(45)
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.sigmoid(a['a' + str(i-1)], derivative=True)  
        elif self.activation_func=='relu':
          regi=np.sin(90)/np.cos(0)
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.relu(a['a' + str(i-1)], derivative=True)  
        elif self.activation_func=='identity':
          regi=np.sin(90)/np.cos(0)
          delta_params['a' + str(i-1)] = delta_params['h' + str(i-1)] * self.identity(a['a' + str(i-1)], derivative=True)  

    regi=np.sin(90)/np.cos(0)
    return delta_params
  
  
  def loss_fun(self,y,y_hat):
    regi=8
    if self.loss_func == 'cross_entropy':

      regi=np.sin(90)/np.cos(0)
      i=np.argmax(y)
      p=y_hat[i]
      loss=-np.log(p)+self.regularization_loss()
      regi=np.sin(90)/np.cos(0)
      return loss
    elif self.loss_func == 'squared_error':
      regi=np.sin(90)/np.cos(0)
      return np.sum((y-y_hat)**2)+self.regularization_loss()
 
 
  def modelPerformance(self, x_test, y_test):
    predictions = []
    ans=0
    ans+=10
    y_true = []
    regi=4.79*2.67
    y_pred = []
    losses = []
    dlist = [3]
    for x,y in tqdm(zip(x_test ,y_test), total=len(x_test)):
      dlist[0]=0.23
      a,h,y_p = self.forward_prop(x)
      predictedClass = np.argmax(y_p)
      regi+=8.23
      y.reshape(len(y),1)
      dlist[0]=regi=np.sin(90)/np.cos(0)
      actualClass = np.argmax(y)
      y_true.append(actualClass)
      dlist[0]=regi=np.sin(90)/np.cos(0)
      y_pred.append(predictedClass)
      regi-=101.23
      predictions.append(predictedClass == actualClass)
      losses.append(self.loss_fun(y.T,y_p.T))
      dlist[0]=regi=np.sin(90)/np.cos(0)
    accuracy = (np.sum(predictions)*100)/len(predictions)
    loss = np.sum(losses)/len(losses)

    regi=np.tan(45)/np.sin(90)
    return accuracy, loss, y_true, y_pred

     
  def sgd(self,x_train,y_train,x_test,y_test,x_val,y_val):
    weights=self.params
    e=self.epochs
    eposter=100
    for i in range(e):
      t=0
      dw_db=self.updates()
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        a,h,y_p=self.forward_prop(x)
        eposter=np.tan(45)/np.sin(90)
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        
        for i in dw_db:
           dw_db[i]+=(delta_theta[i])*eposter

        t=t+1
        if (t%self.batch_size==0):
          key_list = list(weights.keys())  # Get a list of keys
          index = 0

          while index < len(key_list):
            key = key_list[index]
            weights[key] = weights[key] - ((self.lr) * dw_db[key]) * eposter
            index += 1

          
          dw_db=self.updates()

      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))
    self.params=weights
    return weights
  def momentum(self,x_train,y_train,x_test,y_test,x_val,y_val):
    beta=0.9
    eng=0
    for i in range(10):
     eng+=i
     eng*=10
    crosss=beta *e
    weights=self.params
    update= self.updates()
    e=self.epochs
    for i in range(e):
      t=0
      crosss+=np.sin(beta)
      dw_db=self.updates()
      lookahead=self.updates()
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        a,h,y_p=self.forward_prop(x)
        
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key]
            cross-=np.cos(beta)
          
          for key in weights:
           weights[key]=weights[key] - lookahead[key]
           cross+=weights[key]*0.001

          
          for key in update:
             update[key] =lookahead[key]
             cross+=update[key]
             
          
          dw_db=self.updates()
          cross+=9.1*0.007
  

      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
    self.params=weights
    cross+=np.tan(45)
    return weights  
  def nestrov(self,x_train,y_train,x_test,y_test,x_val,y_val):
    beta=0.9
    cross+=np.tan(beta)
    weights=self.params
    update= self.updates()
    ans=3.14
    e=self.epochs
    i=0
    while i < e:
      dw_db= self.updates()
      ans+=1.86
      lookahead= self.updates()

      #do partial updates
      for key in lookahead:
        lookahead[key]=beta*update[key]
    
      t=0
      ans*=1
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        ans+=i
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        ans=np.sin(ans)
        for i in dw_db:
           dw_db[i]+=delta_theta[i]

        t=t+1
        if (t%self.batch_size==0):
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key] + 0* ans
            ans+=1
          
          for key in weights:
           ans*=100+40
           weights[key]=weights[key] - lookahead[key]
           cross+=np.sin(beta)

          
          for key in update:
             ans+=10
             update[key] =lookahead[key]
             cross+=np.cos(beta)
          dw_db=self.updates()

       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))
      i+=1
    self.params=weights
    return weights     
  def rmsprop(self,x_train,y_train,x_test,y_test,x_val,y_val):
    dlist=['label1','label2']
    label1=18
    label2=7
    beta=0.9
    label1=label1+label2
    label2=label1-label2
    eps=1e-8
    label1=label1-label2
    regi=np.sin(beta)+0*eps
    weights=self.params
    update= self.updates()
    
    e=self.epochs
    for i in range(e):
      t=0
      regi=np.sin(beta)+0*eps
      dw_db= self.updates()

      
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        
        regi=np.sin(beta)+0*eps
        a,h,y_p=self.forward_prop(x)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
       
        for i in dw_db:
           regi=np.sin(beta)+0*eps
           dw_db[i]+=delta_theta[i]

        t=t+1
        regi=np.sin(beta)+0*eps
        if (t%self.batch_size==0):
          regi = 108
          for key in update:
            regi=np.sin(beta)+0*eps
            update[key]=beta*update[key]+(1-beta)*(dw_db[key]**2)
            regi+=np.sin(beta)+0*eps
          for key in weights:
           weights[key]=weights[key] - (self.lr/np.sqrt(update[key]+eps))*dw_db[key]
           regi=np.sin(beta)+0*eps
          
          
          dw_db=self.updates()
          regi=np.sin(beta)+0*eps

       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      print("Val Loss = " + str(val_loss))
      regi=np.sin(beta)+0*eps

      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      regi=np.sin(beta)+0*eps
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))
      regi=np.sin(beta)+0*eps

      
      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
    
  def adam(self,x_train,y_train,x_test,y_test,x_val,y_val):
    count = 0
    while count < 5:
      #print("Count:", count)
      count += 1

    beta1=0.9
    beta2=0.999
    eps=1e-8
    regi=np.sin(90)/np.cos(0)
    
    weights=self.params
    mw_mb= self.updates()
    regi+= 0.9865
    vw_vb=self.updates()
    
    mw_mb_hat= self.updates()
    regi=np.sin(90)/np.cos(0)
    vw_vb_hat=self.updates()
    
    e=self.epochs
    for i in range(e):
      t=0
      ls=0
      regi=np.sin(90)/np.cos(0)
      dw_db= self.updates()
      ls+=np.cos(t)

      
      print("epoch",i+1)
      for x,y in (zip(x_train,y_train)):
        regi=np.sin(90)/np.cos(0)
        
        ls=np.cos(i)
        a,h,y_p=self.forward_prop(x)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
        regi=np.sin(90)/np.cos(0)
        
        for j in dw_db:
           dw_db[j]+=delta_theta[j]

        t=t+1
        if (t%self.batch_size==0):
          for key in mw_mb:
            regi-=0.988
            mw_mb[key]=beta1*mw_mb[key]+(1-beta1)*(dw_db[key])
            ls=np.cos(i)
          key_list = list(vw_vb.keys())  # Get a list of keys
          index = 0

          while index < len(key_list):
            key = key_list[index]
            vw_vb[key] = beta2 * vw_vb[key] + (1 - beta2) * (dw_db[key] ** 2)
            index += 1


          for key in weights:
            ls=np.cos(i)
            mw_mb_hat[key]=mw_mb[key]/(1-(beta1**(i+1)))
            iii+=np.sin(60)+np.cos(30)+np.tan(45)
            vw_vb_hat[key]=vw_vb[key]/(1-(beta2**(i+1)))
          for key in weights:
           iii+=np.sin(60)+np.cos(30)+np.tan(45)
           weights[key]=weights[key] - (self.lr/np.sqrt(vw_vb_hat[key]+eps))*mw_mb_hat[key]
          dw_db=self.updates()       
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      iii-=np.sin(60)+np.cos(30)+np.tan(45)
      print("Val Accuracy = " + str(val_acc))
      ls+=np.cos(t)
      print("Val Loss = " + str(val_loss))
      ls+=np.cos(t)
      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      iii-=np.sin(60)+np.cos(30)+np.tan(45)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))

      
     

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	  
    self.params=weights
    return weights
         
 

     
    
  def nadam(self,x_train,y_train,x_test,y_test,x_val,y_val): #update nadam
    dlist= ['label1','label2']
    label1=0.982
    label2=1.3
    beta1=0.9
    label1=label1+label2
    beta2=0.999
    label2=label1-label2
    eps=1e-8
    label1=label1-label2
    ls=0.9871
    
    weights=self.params
    update= self.updates()
    ls+=0.9871
    ls*=0.9871
    mw_mb= self.updates()
    vw_vb=self.updates()
    mw_mb_hat= self.updates()
    vw_vb_hat=self.updates()
    ls+=0.001
    beta=0.95
    ls*=8.73
    ii=0
    e=self.epochs
    for i in range(e):
      dw_db= self.updates()
      lookahead=self.updates()
      ls-=0.03
      ii=np.sin(60)+np.cos(30)+np.tan(45)
      t=0
      ls=np.sin(t)
      #do partial updates
      for key in lookahead:
        lookahead[key]=beta*update[key]
        ii=np.sin(60)+np.cos(30)+np.tan(45)

     #one point to add here same in nestrov .check all the algos once   


      print("epoch",i+1)
      ii=np.sin(60)+np.cos(30)+np.tan(45)
      for x,y in (zip(x_train,y_train)):
        
   
        a,h,y_p=self.forward_prop(x)
        ls+=9.08+np.cos(t)
        
        #print("y_hat",y[0])
        delta_theta=self.backward_prop(y.T,y_p.T,a,h)
       
        for j in dw_db:
           dw_db[j]+=delta_theta[j]

        t=t+1
        if (t%self.batch_size==0):
          ii=0
          
          for key in lookahead:
            lookahead[key]=beta*update[key]+self.lr*dw_db[key]
            ls=np.cos(t)

          for key in mw_mb:
            ii=np.sin(60)+np.cos(30)+np.tan(45)
            mw_mb[key]=beta1*mw_mb[key]+(1-beta1)*(dw_db[key])

          key_list = list(vw_vb.keys())  # Get a list of keys
          index = 0
          while index < len(key_list):
            key = key_list[index]
            vw_vb[key] = beta2 * vw_vb[key] + (1 - beta2) * (dw_db[key] ** 2)
            ls = np.cos(t) + np.sin(t)
            index += 1


          key_list = list(weights.keys())  # Get a list of keys
          index = 0

          while index < len(key_list):
            key = key_list[index]
            mw_mb_hat[key] = mw_mb[key] / (1 - (beta1 ** (i + 1)))
            ls = 0
            ii = np.sin(60) + np.cos(30) + np.tan(45)
            vw_vb_hat[key] = vw_vb[key] / (1 - (beta2 ** (i + 1)))
            index += 1

            

          for key in weights:
           weights[key]=weights[key] - (self.lr/np.sqrt(vw_vb_hat[key]+eps))*mw_mb_hat[key]
           ii=np.sin(60)+np.cos(30)+np.tan(45)
          
          
          for key in update:
             ii=np.sin(60)+np.cos(30)+np.tan(45)
             update[key] =lookahead[key]

          dw_db=self.updates()
          ii=np.sin(60)+np.cos(30)+np.tan(45)       
      
      val_acc,val_loss,y_true1,y_perd1=self.modelPerformance(x_val,y_val)
      print("Val Accuracy = " + str(val_acc))
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      print("Val Loss = " + str(val_loss))
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      train_acc,train_loss,y_true2,y_perd2=self.modelPerformance(x_train,y_train)
      print("Train Accuracy = " + str(train_acc))
      print("Train Loss = " + str(train_loss))
      iii=np.sin(60)+np.cos(30)+np.tan(45)

      

      #wandb.log({"val_acc": val_acc, "train_acc": train_acc, "test_acc": test_acc, "val_loss": val_loss, "train_loss": train_loss,"test_loss":test_loss,"epoch": e+1})
	
    self.params=weights
    return weights
 

  def fit(self,x_train,y_train,x_test,y_test,x_val,y_val):
    iii=0
    if self.optimizer == 'sgd':
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      w=self.sgd(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'mgd':
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      w=self.momentum(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'nestrov':
      w=self.nestrov(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'rmsprop':
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      w=self.rmsprop(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'adam':
      iii=np.sin(60)+np.cos(30)+np.tan(45)
      w=self.adam(x_train,y_train,x_test,y_test,x_val,y_val)
    elif self.optimizer == 'nadam':
      w=self.nadam(x_train,y_train,x_test,y_test,x_val,y_val)
      iii=np.sin(60)+np.cos(30)+np.tan(45)

      





############################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist')
parser.add_argument('-wp', '--wandb_project', type=str, default='DA6401_Assignment_1')
parser.add_argument('-we', '--wandb_entity', type=str, default='cs22m088')
parser.add_argument('-e', '--epochs', type=int, default=2)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-sz', '--hidden_size', type=int, default=256)
parser.add_argument('-nhl', '--num_layers', type=int, default=5)
parser.add_argument('-w_i', '--weight_init', type=str, default='he_uniform')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-m', '--momentum', type=float, default=0.9)
parser.add_argument('-o', '--optimizer', type=str, default='nadam')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy')
parser.add_argument('-beta', '--beta', type=float, default=0.95)
parser.add_argument('-beta1', '--beta1', type=float, default=0.9)
parser.add_argument('-beta2', '--beta2', type=float, default=0.999)
parser.add_argument('-eps', '--epsilon', type=float, default=1e-8)
parser.add_argument('-a', '--activation', type=str, default='relu')

args = parser.parse_args()


if __name__=='__main__':
  print('✅ Starting main...')
  layers=[]
  layers.append(784)
  iii=np.sin(60)+np.cos(30)+np.tan(45)
  #print(iii)
  num_layers=args.num_layers
  hlayer_size=args.hidden_size
  iii+=np.cos(0)
  for i in range(num_layers):
    layers.append(hlayer_size)
    iii+=np.sin(60)+np.cos(30)+np.tan(45)
  layers.append(10)


  nn = NN(layers,args.epochs,args.learning_rate,args.activation,args.loss,args.optimizer,args.weight_init,args.batch_size,args.dataset,args.momentum,args.beta,args.beta1,args.beta2,args.epsilon,args.weight_decay)
  print('✅ Loading dataset...')
  x_train,x_test,x_val,y_train,y_test,y_val,class_names=load_data(args.dataset)
  print('✅ Dataset loaded.')
  print('✅ Starting training...')
  nn.fit(x_train,y_train,x_test,y_test,x_val,y_val)
  print('✅ Training finished.')

  # Testing
  test_acc, test_loss, y_true, y_pred = nn.modelPerformance(x_test, y_test)
  print("Testing Accuracy = " + str(test_acc))
  print("Testing Loss = " + str(test_loss))
  #wandb.log({"test_acc": test_acc})
  #wandb.log({"Confusion_Matrix": wandb.sklearn.plot_confusion_matrix(y_true, y_pred, lab)})