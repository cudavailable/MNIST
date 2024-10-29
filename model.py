import torch.nn.functional as F
import torch.nn as nn

class Model(nn.Module):
  def __init__(self, channels, kernels, strides, dims):
    super(Model,self).__init__()
    """
    channels = [1, 6, 16]
    kernels = [5, 2]
    strides = [1, 2]
    dims = [256, 120, 84, 10]
    """
    self.conv1 = nn.Conv2d(in_channels=channels[0], out_channels=channels[1],
                           kernel_size=kernels[0], stride=strides[0])
    self.conv2 = nn.Conv2d(in_channels=channels[1], out_channels=channels[2],
                           kernel_size=kernels[0], stride=strides[0])
    self.pool = nn.MaxPool2d(kernel_size=kernels[1], stride=strides[1])
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(dims[0], dims[1])
    self.fc2 = nn.Linear(dims[1], dims[2])
    self.fc3 = nn.Linear(dims[2], dims[3])

  def forward(self, x):
    x = F.relu(self.conv1(x)) # 6*24*24
    x = self.pool(x) # 6*12*12
    x = F.relu(self.conv2(x)) # 16*8*8
    x = self.pool(x) # 16*4*4
    x = self.flatten(x) # 256
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = F.softmax(x, dim=1) # activation function

    return x

