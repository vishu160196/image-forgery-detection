from keras.applications.vgg19 import VGG19
import numpy as np


def main():
    model = VGG19(weights='imagenet', include_top=False)

    x_train=np.load('train_data_reduced/x_train_150000.npy')
    x_cv=np.load('train_data_reduced/x_cv_25000.npy')


    x_train_bottleneck = model.predict(x_train)
    x_cv_bottleneck = model.predict(x_cv)

    np.save('train_data_reduced/x_train_vgg19_150000_bottleneck.npy', x_train_bottleneck)
    np.save('train_data_reduced/x_cv_vgg19_25000_bottleneck.npy', x_cv_bottleneck)



if __name__=='__main__':
    main()