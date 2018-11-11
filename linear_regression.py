import numpy as np
from sklearn.linear_model import LinearRegression
from data import X_train, X_test, Y_train, Y_test

model = LinearRegression()
model.fit(X_train, Y_train)
print('Params:', model.coef_.size)

Y_hat_test = model.predict(X_test)
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
