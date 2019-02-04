import numpy as np
from keras import applications
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
	vgg_model=applications.VGG16(weights='imagenet', include_top=False, input_shape=(64,64,3))

	model_aug=Sequential()
	model_aug.add(vgg_model)

	top_model=Sequential()
	top_model.add(Flatten(input_shape=(2, 2, 512)))
	#model_aug.add(Dropout(0.3))
	top_model.add(Dense(64, activation='relu'))
	
	top_model.add(Dense(1, activation='sigmoid'))
	top_model.load_weights('top_model_full_data_custom_lr_weights.h5')

	model_aug.add(top_model)
	

	for layer in model_aug.layers[0].layers[:17]:
		layer.trainable=False

	model_aug.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-6), metrics=['accuracy'])

	print(model_aug.summary())

	model_aug.fit(x_train, y_train, epochs=35, batch_size=32, validation_data=(x_cv, y_cv), verbose=1)
	model_aug.save_weights('fine_tuned_model_adam_weights.h5')


if __name__=='__main__':
	main()
