import os
import torch
from data_process import getDataLoader
from torch import nn
from logger import Logger
from config import MnistConfig
from model import Model

def train(device, model, optimizer, criterion, train_loader, test_loader):
  """Train and Test"""
  # Train
  train_loss = 0
  train_acc = 0
  model.train()  # set training mode

  for img, target in train_loader:
    img = img.to(device)
    target = target.to(device)
    out = model(img)

    optimizer.zero_grad()  # clear grad
    loss = criterion(out, target)
    loss.backward()  # backward propagation
    optimizer.step()

    train_loss += loss.item()  # record training loss
    _, pred = out.max(1)
    num_correct = (pred == target).sum().item()
    acc = num_correct / img.shape[0]
    train_acc += acc

  # Test
  test_loss = 0
  test_acc = 0
  model.eval()  # set testing mode

  for img, target in test_loader:
    img = img.to(device)
    target = target.to(device)
    out = model(img)

    loss = criterion(out, target)
    test_loss += loss.item()

    _, pred = out.max(1)
    num_correct = (pred == target).sum().item()
    acc = num_correct / img.shape[0]
    test_acc += acc

  return train_loss, train_acc, test_loss, test_acc

def main():
  # preparations for training
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  model = Model(MnistConfig.channels, MnistConfig.kernels, MnistConfig.strides, MnistConfig.dims)
  model = model.to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=MnistConfig.lr)
  criterion = nn.CrossEntropyLoss().to(device)

  best_epoch_info = {'epoch': 0, 'metrics': None}

  # set log path
  log_dir = os.path.join(MnistConfig.workspace, "log")
  if log_dir is not None and not os.path.exists(log_dir):
    os.makedirs(log_dir)
  logger = Logger(os.path.join(log_dir, "log.txt"))
  logger.write(f"Using {device} device\n")  # log

  weights_dir = os.path.join(log_dir, 'weights')
  if weights_dir is not None and not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

  # get dataloaders
  train_loader, test_loader = getDataLoader()

  len_train_dataset = len(train_loader.dataset)
  len_test_dataset = len(test_loader.dataset)
  logger.write(f"Train set size: {len_train_dataset}\n Test set size: {len_test_dataset}\n\n")  # log

  for epoch in range(MnistConfig.num_epoch):
    # return train_loss, train_acc, test_loss, test_acc
    train_loss, train_acc, test_loss, test_acc = train(device, model, optimizer, criterion, train_loader, test_loader)

    train_acc /= len(train_loader)
    test_acc /= len(test_loader)
    train_loss /= len(train_loader)
    test_loss /= len(test_loader)
    # print(f"Epoch[{epoch+1}/{num_epoch}], Train Loss:{train_loss:.4f}, Train Acc:{train_acc:.4f}, Test Loss:{test_loss:.4f}, Test Acc:{test_acc:.4f}")
    logger.write(
      f"Epoch[{epoch + 1}/{MnistConfig.num_epoch}], Train Loss:{train_loss:.4f}, Train Acc:{train_acc:.4f}, Test Loss:{test_loss:.4f}, Test Acc:{test_acc:.4f}\n")  # log

    # Keep recording the params of the best model
    if best_epoch_info['metrics'] is None or test_loss < best_epoch_info['metrics']:
      best_epoch_info['epoch'] = epoch + 1
      best_epoch_info['metrics'] = test_loss
      best_weight_path = os.path.join(weights_dir, "best_model_epoch.pth")
      model.eval()
      torch.save(model.state_dict(), best_weight_path)

  # end of traning
  logger.write("\n\nTraining completed\n\n")
  logger.write(f"Best Epoch: {best_epoch_info['epoch']}\n")
  # close
  logger.close()

if __name__=="__main__":
  main()