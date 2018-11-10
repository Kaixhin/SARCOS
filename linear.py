import numpy as np
from data import train_data, test_data

X_train, Y_train = train_data[:, :21], train_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]
bias_train, bias_test = np.ones((X_train.shape[0], 1)), np.ones((X_test.shape[0], 1))
W = np.linalg.lstsq(np.concatenate((X_train, bias_train), axis=1), Y_train)[0]
print('Params:', W.shape[0] * W.shape[1])
Y_hat_test = np.concatenate((X_test, bias_test), axis=1) @ W
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
