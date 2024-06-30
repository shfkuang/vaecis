import numpy as np
import matplotlib.pyplot as plt
import os

def lines_to_np_array(lines):
    return np.array([[int(i) for i in line.split()] for line in lines])

def get_binarized_MNIST(DATASETS_DIR='./binarymnist', random_seed=0):
    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_train.amat')) as f:
            lines = f.readlines()
    train = lines_to_np_array(lines).astype('float32').reshape(-1,28,28)
        
    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    valid = lines_to_np_array(lines).astype('float32').reshape(-1,28,28)

    with open(os.path.join(DATASETS_DIR, 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test = lines_to_np_array(lines).astype('float32').reshape(-1,28,28)
    return train, valid, test


def create_mcar_mask(n_rows, n_columns, percentage_of_missingness=0.1, seed=0):
    np.random.seed(seed)
    mask = np.ones((n_rows, n_columns))

    for i in range(mask.shape[0]):
        missing_values = np.random.choice(np.arange(0, n_columns, 1), size=int(percentage_of_missingness * n_columns),
                                          replace=False)
        mask[i, missing_values] = 0

    return mask.reshape(-1,28,28)


def create_mar_mask(data):

    masks = np.zeros((data.shape[0], data.shape[1]))
    for i, example in enumerate(data):
        h = (1. / (784. / 2.)) * np.sum(example[int(784 / 2):]) + 0.3
        pi = np.random.binomial(2, h)

        _mask = np.ones(example.shape[0])

        if pi == 0:
            _mask[196:(2 * 196)] = 0

        elif pi == 1:
            _mask[0:(2 * 196)] = 0

        elif pi == 2:
            _mask[0:196] = 0

        else:
            print('There is a problem mate, the pub is close')
            break

        masks[i, :] = _mask

    return masks.reshape(-1,28,28)

def get_imputation(data, observed_mask, missing_mask, imputation_value):
    return data * observed_mask + missing_mask * imputation_value

