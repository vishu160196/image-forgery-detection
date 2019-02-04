from keras import applications
import numpy as np


def main():
    model = applications.VGG16(weights='imagenet', include_top=False)

    x_train=np.load('train_data/x_train.npy')[:1]
    x_cv=np.load('train_data/x_cv.npy')[:1]


    x_train_bottleneck = model.predict(x_train)
    x_cv_bottleneck = model.predict(x_cv)

    # np.save('train_data/x_train_bottleneck.npy', x_train_bottleneck)
    # np.save('train_data/x_cv_bottleneck.npy', x_cv_bottleneck)

    print(x_train_bottleneck[0].shape)


if __name__=='__main__':
    main()
