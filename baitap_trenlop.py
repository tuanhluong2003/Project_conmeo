import numpy as np
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model


data = pd.read_csv("USA_Housing.csv")
dt_Train, dt_test = train_test_split(data, test_size=0.1, shuffle=False)

X_train = dt_Train.iloc[:,:5]
Y_train = dt_Train.iloc[:,5]
X_test = dt_test.iloc[:,:5]
Y_test = dt_test.iloc[:,5]


reg_lasso = linear_model.Lasso(alpha=1)
reg_lasso.fit(X_train,Y_train)
y_pred_lasso = reg_lasso.predict(X_test)

reg_ridge = linear_model.Ridge(alpha=1)
reg_ridge.fit(X_train,Y_train)
y_pred_ridge = reg_ridge.predict(X_test)


reg_linear = LinearRegression().fit(X_train, Y_train)#kiem tra nha ve nha nho lam
y_pred_linear = reg_linear.predict(X_test)
y = np.array(Y_test)

print("linear : %.9f" % r2_score(Y_test,y_pred_linear))
print("lasso : %.9f" % r2_score(Y_test,y_pred_lasso))
print("ridge : %.9f" % r2_score(Y_test,y_pred_ridge))


print("Thuc te \t du doan \t chech lech")
for i in range(0,len(y)):
    print("%.2f\t" %abs(y[i]-y_pred_linear[i]), "%.2f\t" %abs(y[i]-y_pred_lasso[i]),"%.2f" %abs(y[i]-y_pred_ridge[i]))