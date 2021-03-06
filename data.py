import numpy as np
from scipy.io import loadmat

train_data = loadmat('sarcos_inv.mat')['sarcos_inv'].astype(np.float32)
val_data, train_data = train_data[:4448], train_data[4484:].astype(np.float32)
test_data = loadmat('sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float32)

X_train, Y_train = train_data[:, :21], train_data[:, 21:]
X_val, Y_val = val_data[:, :21], val_data[:, 21:]
X_test, Y_test = test_data[:, :21], test_data[:, 21:]
