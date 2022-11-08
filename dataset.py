import typing as tp

import numpy as np
import scipy.io as io


def load_data_mat(filename, max_samples, seed=42):
    '''
    Loads numpy arrays from .mat file

    Returns:
    X, np array (num_samples, 32, 32, 3) - images
    y, np array of int (num_samples) - labels
    '''
    raw: dict = io.loadmat(filename)
    X: np.ndarray = raw['X']  # Array of [32, 32, 3, n_samples]
    y: np.ndarray = raw['y']  # Array of [n_samples, 1]

    X: np.ndarray = np.moveaxis(X, [3], [0])
    y: np.ndarray = y.flatten()
    # Fix up class 0 to be 0
    y[y == 10] = 0

    np.random.seed(seed)
    samples = np.random.choice(np.arange(X.shape[0]),
                               max_samples,
                               replace=False)

    return X[samples].astype(np.float32), y[samples]


def extension_to_vector(data: np.ndarray) -> np.ndarray:
    individual_objects = data.shape[0]
    pixsel_on_object = int(data.size / individual_objects)
    out = np.zeros((individual_objects, pixsel_on_object), dtype=np.float32)
    
    for i, obj in enumerate(data):
        out[i] = obj.flatten()
    return out


def disclosure_to_vector(train:np.ndarray)-> np.ndarray:
    out = []
    for i in train:
        out.append(vector_representation(i, 10))
    return np.array(out, dtype=int)


def vector_representation(val, max_val:int=10) -> np.ndarray:
    out = np.zeros(max_val)
    out[val] = 1
    return out

