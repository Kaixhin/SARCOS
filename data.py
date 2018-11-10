import numpy as np
from scipy.io import loadmat

train_data = loadmat('sarcos_inv.mat')['sarcos_inv'].astype(np.float32)
val_data, train_data = train_data[:4448], train_data[4484:].astype(np.float32)
test_data = loadmat('sarcos_inv_test.mat')['sarcos_inv_test'].astype(np.float32)
