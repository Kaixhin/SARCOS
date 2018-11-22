import argparse
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from data import X_train, X_test, Y_train, Y_test
from utils import get_tree_param_counts

parser = argparse.ArgumentParser(description='SARCOS Gradient Boosted Trees')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators')
parser.add_argument('--max-depth', type=int, default=10, help='Max depth')
args = parser.parse_args()

model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=args.seed), n_jobs=-1)
model.fit(X_train, Y_train)
param_counts = [[get_tree_param_counts(m_[0]) for m_ in m.estimators_] for m in model.estimators_]
param_counts = list(zip(*[item for sublist in param_counts for item in sublist]))
print('Param Counts:', (sum(param_counts[0]), sum(param_counts[1])))

Y_hat_test = model.predict(X_test)
MSE = np.square(Y_test - Y_hat_test).mean()
print('Test MSE:', MSE)
