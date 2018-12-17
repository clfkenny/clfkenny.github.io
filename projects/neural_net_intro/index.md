---
layout: projects
title: Shallow Net
author: Kenny Lov
---

## Introduction


```python
import numpy
from sklearn import datasets, preprocessing
X = datasets.load_boston().data
X = preprocessing.normalize(X)
y = datasets.load_boston().target
```


```python
class ShallowNet:
    
    def __init__(self, params):
        # collect desired size of layers
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        
        # initialize weights
        self.W1 = np.random.randn(self.hidden_size, self.input_size) # (hid_s, feature_s)
        self.b1 = np.zeros((self.hidden_size,1)) # (hid_s, 1)
        
        self.W2 = np.random.randn(self.output_size, self.hidden_size) # (out_s, hid_s)
        self.b2 = np.zeros((self.output_size, 1)) # (out_s, 1)
    
    def sigmoid(self, x): # sigmoid activation function
        return 1/(1+np.exp(-x))
    
    def sigmoid_p(self, x): # derivative of sigmoid function
        return self.sigmoid(x) * (1-self.sigmoid(x))

    def forward_prop(self, X):
        self.Z1 = self.W1.dot(X.T) + self.b1 # (hid_s, feature_s) x (feature_s, sample_s)
        self.A1 = self.sigmoid(self.Z1) # (hid_s, sample_s)
        self.out = self.W2.dot(self.A1) + self.b2 #(out_size, hid_size) x (hid_s, sample_s)

    def mse(self, y): # calculate cost
        return np.sum((self.out-y)**2) / len(y)
    
    def backward_prop(self, X, y, lr):
        # calculate gradients with respect to loss
        self.d_out = self.out - y
        self.dW2 = (1/self.input_size) * self.d_out.dot(self.A1.T)
        self.db2 = (1/self.input_size) * np.sum(self.d_out, axis=1, keepdims = True)
        
        self.dZ1 = self.W2.T.dot(self.d_out) * self.sigmoid_p(self.Z1)
        self.dW1 = (1/self.input_size) * self.dZ1.dot(X)
        self.db1 = (1/self.input_size) * np.sum(self.dZ1, axis=1, keepdims= True)
        
        # update weights using calculated gradients
        self.W2 -= lr*self.dW2
        self.b2 -= lr*self.db2
        self.W1 -= lr*self.dW1
        self.b1 -= lr*self.db1
    
    def one_epoch(self, X, y, lr): # one epoch: forward and back prop 
        self.forward_prop(X)
        self.backward_prop(X, y, lr)
        
    def predict(self, X_pred):
        Z1 = self.W1.dot(X_pred.T) + self.b1
        A1 = self.sigmoid(Z1)
        out = self.W2.dot(A1) + self.b2
        return out
        
```


```python
epochs = 100
lr = 0.001
hid_size = 20

net = ShallowNet({"input_size": X.shape[1], "hidden_size": hid_size, 'output_size': 1})
for i in range(epochs):
    net.one_epoch(X, y, lr)
    print('Epoch: {} - MSE: {}'.format(i+1, net.mse(y)))
```

    Epoch: 1 - MSE: 507.02439374794574
    Epoch: 2 - MSE: 282.77338655994
    Epoch: 3 - MSE: 165.6615922825101
    Epoch: 4 - MSE: 112.41550337217129
    Epoch: 5 - MSE: 92.58335428539773
    Epoch: 6 - MSE: 86.10414178882516
    Epoch: 7 - MSE: 84.01902516307435
    Epoch: 8 - MSE: 83.21886306079644
    Epoch: 9 - MSE: 82.77185105947386
    Epoch: 10 - MSE: 82.4204931054145
    Epoch: 11 - MSE: 82.09660457442592
    Epoch: 12 - MSE: 81.78246610995421
    Epoch: 13 - MSE: 81.47344686658035
    Epoch: 14 - MSE: 81.16826324811854
    Epoch: 15 - MSE: 80.86648625917047
    Epoch: 16 - MSE: 80.56791161239583
    Epoch: 17 - MSE: 80.27240226885233
    Epoch: 18 - MSE: 79.9798493649269
    Epoch: 19 - MSE: 79.6901623173241
    Epoch: 20 - MSE: 79.40326590619092
    Epoch: 21 - MSE: 79.11909892369137
    Epoch: 22 - MSE: 78.83761312144254
    Epoch: 23 - MSE: 78.55877218831434
    Epoch: 24 - MSE: 78.28255071810561
    Epoch: 25 - MSE: 78.00893317263278
    Epoch: 26 - MSE: 77.73791285187438
    Epoch: 27 - MSE: 77.4694908816967
    Epoch: 28 - MSE: 77.20367522804331
    Epoch: 29 - MSE: 76.94047974523933
    Epoch: 30 - MSE: 76.67992326511123
    Epoch: 31 - MSE: 76.42202873275933
    Epoch: 32 - MSE: 76.16682239392831
    Epoch: 33 - MSE: 75.91433303795645
    Epoch: 34 - MSE: 75.6645912992527
    Epoch: 35 - MSE: 75.41762901917755
    Epoch: 36 - MSE: 75.17347866912553
    Epoch: 37 - MSE: 74.93217283456828
    Epoch: 38 - MSE: 74.69374375884941
    Epoch: 39 - MSE: 74.45822294466248
    Epoch: 40 - MSE: 74.22564081041173
    Epoch: 41 - MSE: 73.99602639806861
    Epoch: 42 - MSE: 73.76940712870001
    Epoch: 43 - MSE: 73.54580860155907
    Epoch: 44 - MSE: 73.325254432482
    Epoch: 45 - MSE: 73.10776612731811
    Epoch: 46 - MSE: 72.89336298621271
    Epoch: 47 - MSE: 72.68206203474654
    Epoch: 48 - MSE: 72.47387797818985
    Epoch: 49 - MSE: 72.26882317543557
    Epoch: 50 - MSE: 72.06690762951219
    Epoch: 51 - MSE: 71.8681389919304
    Epoch: 52 - MSE: 71.6725225784704
    Epoch: 53 - MSE: 71.48006139435888
    Epoch: 54 - MSE: 71.29075616710705
    Epoch: 55 - MSE: 71.1046053855763
    Epoch: 56 - MSE: 70.92160534410277
    Epoch: 57 - MSE: 70.74175019074306
    Epoch: 58 - MSE: 70.56503197890149
    Epoch: 59 - MSE: 70.39144072176384
    Epoch: 60 - MSE: 70.2209644490971
    Epoch: 61 - MSE: 70.0535892660793
    Epoch: 62 - MSE: 69.8892994139057
    Epoch: 63 - MSE: 69.72807733197467
    Epoch: 64 - MSE: 69.56990372149795
    Epoch: 65 - MSE: 69.41475761040444
    Epoch: 66 - MSE: 69.26261641941986
    Epoch: 67 - MSE: 69.11345602920854
    Epoch: 68 - MSE: 68.9672508484608
    Epoch: 69 - MSE: 68.82397388280152
    Epoch: 70 - MSE: 68.6835968043871
    Epoch: 71 - MSE: 68.54609002204495
    Epoch: 72 - MSE: 68.41142275180135
    Epoch: 73 - MSE: 68.27956308763119
    Epoch: 74 - MSE: 68.15047807225778
    Epoch: 75 - MSE: 68.02413376782256
    Epoch: 76 - MSE: 67.9004953262434
    Epoch: 77 - MSE: 67.7795270590776
    Epoch: 78 - MSE: 67.66119250670846
    Epoch: 79 - MSE: 67.54545450667794
    Epoch: 80 - MSE: 67.43227526099395
    Epoch: 81 - MSE: 67.3216164022494
    Epoch: 82 - MSE: 67.21343905839956
    Epoch: 83 - MSE: 67.1077039160556
    Epoch: 84 - MSE: 67.00437128216444
    Epoch: 85 - MSE: 66.90340114395785
    Epoch: 86 - MSE: 66.80475322706737
    Epoch: 87 - MSE: 66.70838705171533
    Epoch: 88 - MSE: 66.61426198690594
    Epoch: 89 - MSE: 66.52233730255395
    Epoch: 90 - MSE: 66.43257221950157
    Epoch: 91 - MSE: 66.34492595738729
    Epoch: 92 - MSE: 66.25935778034197
    Epoch: 93 - MSE: 66.17582704049902
    Epoch: 94 - MSE: 66.09429321931631
    Epoch: 95 - MSE: 66.0147159667172
    Epoch: 96 - MSE: 65.93705513806646
    Epoch: 97 - MSE: 65.8612708290061
    Epoch: 98 - MSE: 65.78732340818196
    Epoch: 99 - MSE: 65.71517354789951
    Epoch: 100 - MSE: 65.64478225275236

