import numpy as np

tmpx = []
tmpy = []
index = 1
with open('input.txt') as rf:
    line = rf.readline()
    while line:
        tmpline = [float(i) for i in line.split()]
        tmpx.append(tmpline[:-1])
        tmpy.append([tmpline[-1]])
        index += 1
        line = rf.readline()
X = np.array(tmpx)
X_train = np.insert(X,0,1,axis = 1).T
Y_train = np.array(tmpy)

w = np.linalg.pinv(X_train@X_train.T)@X_train@Y_train
print(X_train.T@w)