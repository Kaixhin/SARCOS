import numpy as np
from data import train_data, test_data

X_train, Y_train = train_data[:, :21], train_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]
W = np.linalg.lstsq(X_train, Y_train)[0]
Y_hat_test = X_test @ W
MSE = np.square(Y_test - Y_hat_test).mean()
print(MSE)
