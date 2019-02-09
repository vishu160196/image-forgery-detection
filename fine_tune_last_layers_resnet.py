import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras import optimizers


def main():
	x_train=np.load('train_data/x_train.npy')
	x_cv=np.load('train_data/x_cv.npy')

	y_train=np.load('train_data/y_train.npy')
	y_cv=np.load('train_data/y_cv.npy')

	# load VGG16
	resnet_model=ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

	model_aug=Sequential()
	model_aug.add(resnet_model)

	top_model=Sequential()
	top_model.add(Flatten(input_shape=(2, 2, 2048)))
	
	top_model.add(Dense(64, activation='relu'))
	# model_aug.add(Dropout(0.2))

	top_model.add(Dense(1, activation='sigmoid'))
	top_model.load_weights('top_model_inception_64_adam_custom_lr_full_data.h5')

	model_aug.add(top_model)
	

	for layer in model_aug.layers[0].layers[:171]:
		layer.trainable=False

	model_aug.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['accuracy'])

	print(model_aug.summary())

	model_aug.fit(x_train, y_train, epochs=35, batch_size=32, validation_data=(x_cv, y_cv), verbose=1)
	model_aug.save_weights('fine_tuned_model_inception_64_adam_weights.h5')
	# print(model_aug.evaluate(x_cv, y_cv))


if __name__=='__main__':
	main()