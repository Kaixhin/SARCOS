import numpy as np
from sklearn.tree import DecisionTreeRegressor
from data import train_data, test_data

X_train, Y_train = train_data[:, :21], train_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]

model = DecisionTreeRegressor(max_depth=30, random_state=123)
model.fit(X_train, Y_train)

Y_hat_test = model.predict(X_test)
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
