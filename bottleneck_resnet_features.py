from keras import applications
import numpy as np


def main():
    model = applications.resnet50.ResNet50(include_top=False, weights='imagenet')

    x_train=np.load('train_data/x_train.npy')
    x_cv=np.load('train_data/x_cv.npy')

    x_train_bottleneck = model.predict(x_train)
    x_cv_bottleneck = model.predict(x_cv)

    np.save('x_train_resnet_bottleneck.npy', x_train_bottleneck)
    np.save('x_cv_resnet_bottleneck.npy', x_cv_bottleneck)


if __name__=='__main__':
    main()
