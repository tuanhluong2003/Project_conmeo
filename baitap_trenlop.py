import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import math
data = pd.read_csv("USA_Housing.csv")
dt_Train, dt_test = train_test_split(data, test_size=0.3, shuffle=False)

X_train = dt_Train.iloc[:,:5]
Y_train = dt_Train.iloc[:,5]
X_test = dt_test.iloc[:,:5]
Y_test = dt_test.iloc[:,5]


reg = LinearRegression().fit(X_train, Y_train)#kiem tra nha ve nha nho lam
y_pred = reg.predict(X_test)
y = np.array(Y_test)

print("Coeffcient of determination : %.2f" % r2_score(Y_test,y_pred))
print("Thuc te du doan chech lech")
for i in range(0,len(y)):
    print("%.2f" %y[i]," ",y_pred[i]," ",abs(y[i]-y_pred[i]))