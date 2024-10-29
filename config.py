import os

class MnistConfig:
    workspace = os.path.dirname(os.path.abspath(__file__))
    num_epoch = 20
    train_batch_size = 64
    test_batch_size = 128
    channels = [1, 6, 16]
    kernels = [5, 2]
    strides = [1, 2]
    dims = [256, 120, 84, 10]
    lr = 0.001
