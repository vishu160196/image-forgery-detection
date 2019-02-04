import numpy as np
import pickle
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten
import sys
import os


def main():

    train_sizes = np.arange(50e3, 250e3, 25e3, dtype=int)
    cv_sizes = np.arange(25e3, 125e3, 25e3, dtype=int)

    train_size=int(sys.argv[1])
    cv_size=int(sys.argv[2])

    use_full=bool(int(sys.argv[3]))

    if not use_full and (train_size not in train_sizes or cv_size not in cv_sizes):
        print('Invalid size fed')
        exit(1)

    batch_size=32
    epochs=150

    if use_full:
        path='train_data/'
    else:
        path='train_data_reduced/'



    #path_binary='/data/My Drive/Colab Notebooks/image forgery detection/k64 binary 25percent stride8/train_data/'
    #x_train_grayscale=np.load(path_grayscale+'x_train.npy')
    #x_cv_grayscale=np.load(path_grayscale+'x_cv.npy')

    #y_train_grayscale=np.load(path_grayscale+'y_train.npy')
    #y_cv_grayscale=np.load(path_grayscale+'y_cv.npy')

    if use_full:
        x_train=np.load(path+'x_train.npy')
        x_cv=np.load(path+'x_cv.npy')

        y_train=np.load(path+'y_train.npy')
        y_cv=np.load(path+'y_cv.npy')

    else:
        x_train = np.load(path + 'x_train_'+str(train_size)+'.npy')
        x_cv = np.load(path + 'x_cv_'+str(cv_size)+'.npy')

        y_train = np.load(path + 'y_train_'+str(train_size)+'.npy')
        y_cv = np.load(path + 'y_cv_'+str(cv_size)+'.npy')

    # Normalise
    x_train = x_train/255
    x_cv = x_cv/255


    cnn_model=Sequential()

    cnn_model.add(Conv2D(input_shape=(64, 64, 3), filters=20, kernel_size=4, strides=2, padding='valid',
                         activation='relu',  data_format='channels_last'))

    cnn_model.add(Conv2D(filters=15, kernel_size=3, strides=1, padding='valid', activation='relu',
                          data_format='channels_last'))

    cnn_model.add(MaxPool2D(pool_size=3, data_format='channels_last'))

    cnn_model.add(Conv2D(filters=20, kernel_size=4, strides=2, padding='valid', activation='relu',
                          data_format='channels_last'))

    cnn_model.add(MaxPool2D(pool_size=2, data_format='channels_last'))

    # cnn_model.add(Conv2D(filters=15, kernel_size=2, strides=1, padding='valid', activation='relu',
    #                       data_format='channels_last'))

    # cnn_model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu',
    #                       data_format='channels_last'))

    # cnn_model.add(Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu',
    #                      kernel_initializer='he_normal', data_format='channels_last'))

    cnn_model.add(Flatten())

    cnn_model.add(Dropout(0.2))

    cnn_model.add(Dense(1, activation='sigmoid'))

    cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    cnn_model.summary()

    history = cnn_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(x_cv, y_cv))

    cnn_model.save('keras_cnn_model_redone.h5')

if __name__=='__main__':
    main()
