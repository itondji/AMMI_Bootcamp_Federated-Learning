import argparse
from .model import MNIST_2NN
from .transform import iid_partition, mnist_data_train


args = argparse.Namespace(

    rounds = 100
    # client fraction
    C = 0.1
    # number of clients
    K = 100
    # number of training passes on local dataset for each round
    E = 5
    # batch size
    batch_size = 10
    # learning Rate
    lr=0.03
    # dict containing different type of data partition
    data_dict = iid_partition(mnist_data_train, 100)
    # load model
    mnist_mlp = MNIST_2NN()

}