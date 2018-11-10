import argparse
from collections import deque
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from data import train_data, val_data, test_data

parser = argparse.ArgumentParser(description='SARCOS MLP')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--hidden-size', type=int, default=256, help='Hidden size')
parser.add_argument('--layers', type=int, default=5, help='Number of hidden layers')
parser.add_argument('--batch-size', type=int, default=512, help='Minibatch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
args = parser.parse_args()
assert args.layers >= 0
input_size, output_size = 21, 7
torch.manual_seed(args.seed)

X_train, Y_train = torch.from_numpy(train_data[:, :input_size]), torch.from_numpy(train_data[:, input_size:])
X_val, Y_val = torch.from_numpy(val_data[:, :input_size]), torch.from_numpy(val_data[:, input_size:])
X_test, Y_test = torch.from_numpy(test_data[:, :input_size]), torch.from_numpy(test_data[:, input_size:])

train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)
test_dataset = TensorDataset(X_test, Y_test)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

if args.layers > 0:
  layers = [nn.Linear(input_size, args.hidden_size), nn.ReLU()]
  for l in range(args.layers - 1):
    layers += [nn.Linear(args.hidden_size, args.hidden_size), nn.ReLU()]
  layers += [nn.Linear(args.hidden_size, output_size)]
else:
  layers = [nn.Linear(input_size, output_size)]
model = nn.Sequential(*layers)
print('Params:', sum(param.numel() for param in model.parameters()))
optimiser = optim.Adam(model.parameters(), lr=args.lr)
overfit, patience_queue = False, deque([1e10], maxlen=5)

def train(model, optimiser):
  model.train()
  for x, y in train_dataloader:
    optimiser.zero_grad()
    y_hat = model(x)
    loss = F.mse_loss(y_hat, y)
    loss.backward()
    optimiser.step()

def validate(model):
  model.eval()
  validation_loss = 0
  with torch.no_grad():
    for x, y in val_dataloader:
      y_hat = model(x)
      validation_loss += F.mse_loss(y_hat, y, reduction='sum').item()
  return validation_loss / output_size / len(val_dataset)

def test(model):
  model.eval()
  test_loss = 0
  with torch.no_grad():
    for x, y in test_dataloader:
      y_hat = model(x)
      test_loss += F.mse_loss(y_hat, y, reduction='sum').item()
  return test_loss / output_size / len(test_dataset)

while not overfit:
  train(model, optimiser)
  validation_loss = validate(model)
  if validation_loss > max(patience_queue):
    overfit = True
  else:
    patience_queue.append(validation_loss)
  print(validation_loss)

print('Test MSE:', test(model))
