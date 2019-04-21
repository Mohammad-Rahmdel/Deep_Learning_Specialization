"""                      Week3 - Programming Assignment 1
 
"""

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
# dataset[:10]

Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

# print("X.shape:", X.shape)
# print("Y.shape:", Y.shape)
# print("Xoh.shape:", Xoh.shape)
# print("Yoh.shape:", Yoh.shape)


# index = 0
# print("Source date:", dataset[index][0])
# print("Target date:", dataset[index][1])
# print()
# print("Source after preprocessing (indices):", X[index])
# print("Target after preprocessing (indices):", Y[index])
# print()
# print("Source after preprocessing (one-hot):", Xoh[index])
# print("Target after preprocessing (one-hot):", Yoh[index])