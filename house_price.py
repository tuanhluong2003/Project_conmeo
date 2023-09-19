import numpy as np
import pandas as pd

#doc du lieu tu file
data = pd.read_csv('USA_Housing.csv',header=None)
X_train = np.array([data[0], data[1], data[2], data[3], data[4]])

#chen mot cot mot vao hang dau tien
X_train = np.insert(X_train,0,1,axis = 0)
Y_train = np.array([data[5]]).T

X_test = np.array([1, 79545, 5, 7, 4, 23086])
w = np.linalg.pinv(X_train@X_train.T)@X_train@Y_train
print("w = ", w)
print('Giá trị dự đoán mẫu mới: ',X_test@w)
print("Giá trị dự đoán tập huấn luyện:")
print(X_train.T@w)