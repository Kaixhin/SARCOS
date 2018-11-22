import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from data import X_train, X_test, Y_train, Y_test
from utils import get_tree_avg_depth

parser = argparse.ArgumentParser(description='SARCOS Random Forest')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--n-estimators', type=int, default=700, help='Number of estimators')
parser.add_argument('--max-depth', type=int, default=30, help='Max depth')
args = parser.parse_args()

model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed)
model.fit(X_train, Y_train)
print('Param Count:', sum(m.tree_.node_count for m in model.estimators_))
print('Avg. Path Param Count:', sum(get_tree_avg_depth(m) for m in model.estimators_))

Y_hat_test = model.predict(X_test)
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
