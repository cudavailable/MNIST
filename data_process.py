import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from config import MnistConfig

def getDataLoader():
    """Preprocessing MNIST datasets"""
    train_batch_size = MnistConfig.train_batch_size
    test_batch_size = MnistConfig.test_batch_size

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    data_path = os.path.join(MnistConfig.workspace, "data")
    train_dataset = mnist.MNIST(data_path, train=True, transform=transform, download=True)
    test_dataset = mnist.MNIST(data_path, train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader

import matplotlib.pyplot as plt
train_loader, test_loader = getDataLoader()
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib
matplotlib.use('TkAgg')
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth:{}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])

print("picture size : ", example_data[0][0].shape)
plt.show()