import argparse
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from data import X_train, X_test, Y_train, Y_test

parser = argparse.ArgumentParser(description='SARCOS Decision Tree')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--max-depth', type=int, default=30, help='Max depth')
args = parser.parse_args()

model = DecisionTreeRegressor(max_depth=args.max_depth, random_state=args.seed)
model.fit(X_train, Y_train)

Y_hat_test = model.predict(X_test)
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
