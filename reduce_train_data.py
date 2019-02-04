import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm


os.chdir('D:/image forgery detection/k64 binary 25percent stride8/')

x_train=np.load('train_data/x_train.npy')
x_cv=np.load('train_data/x_cv.npy')
y_train=np.load('train_data/y_train.npy')
y_cv=np.load('train_data/y_cv.npy')

train_sizes=[int(50e3), int(75e3), int(100e3), int(125e3), int(150e3), int(175e3), int(200e3), int(225e3)]
cv_sizes=[int(25e3), int(50e3), int(75e3), int(100e3)]

print('generating reduced train sets')
train_reduced=[]
train_reduced_labels=[]
for train_size in tqdm(train_sizes):
    x_train_reduced, x_cv_reduced, y_train_reduced, y_cv_reduced = train_test_split(x_train, y_train, test_size=(x_train.shape[0]-train_size)/x_train.shape[0], stratify=y_train)
    train_reduced.append(x_train_reduced)
    train_reduced_labels.append(y_train_reduced)
print('done')

print('generating reduced cv sets')
cv_reduced=[]
cv_reduced_labels=[]
for cv_size in tqdm(cv_sizes):
    x_cv_reduced, x_test_reduced, y_cv_reduced, y_test_reduced = train_test_split(x_cv, y_cv, test_size=(x_cv.shape[0]-cv_size)/x_cv.shape[0], stratify=y_cv)
    cv_reduced.append(x_cv_reduced)
    cv_reduced_labels.append(y_cv_reduced)
print('done')

os.mkdir('train_data_reduced')

print('saving')
for size, reduced in zip(train_sizes, train_reduced):
    np.save('train_data_reduced/x_train_' + str(size) + '.npy', reduced)

for size, reduced in zip(train_sizes, train_reduced_labels):
    np.save('train_data_reduced/y_train_' + str(size) + '.npy', reduced)

for size, reduced in zip(cv_sizes, cv_reduced):
    np.save('train_data_reduced/x_cv_' + str(size) + '.npy', reduced)

for size, reduced in zip(cv_sizes, cv_reduced_labels):
    np.save('train_data_reduced/y_cv_' + str(size) + '.npy', reduced)
print('done')