import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

#doc du lieu tu file
data = pd.read_csv('USA_Housing.csv',header=None)
X_train = np.array([data[0], data[1], data[2], data[3], data[4]]).T
#chen tu dong mot cot 1 vao cot dau tien
Y_train = np.array([data[5]]).T

reg = LinearRegression().fit(X_train,Y_train)

x_test = np.array([[79545, 5, 7, 4, 23086]])

print("w = ", reg.coef_)
# Sai số
print('w0 = ',reg.intercept_)
print('Giá trị dự đoán mẫu mới: ',reg.predict(x_test))
print("Giá trị dự đoán tập huấn luyện:")
print(reg.predict(X_train))
