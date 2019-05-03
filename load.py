# importing libraries 
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
#import PIL as image
import numpy

img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/test'
nb_train_samples =160
nb_validation_samples = 4
epochs = 100
batch_size = 4

if K.image_data_format() == 'channels_first': 
	input_shape = (3, img_width, img_height) 
else: 
	input_shape = (img_width, img_height, 3) 

model = Sequential() 
model.add(Conv2D(32, (2, 2), input_shape = input_shape)) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(32, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Conv2D(64, (2, 2))) 
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size =(2, 2))) 

model.add(Flatten()) 
model.add(Dense(64)) 
model.add(Activation('relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(1)) 
model.add(Activation('sigmoid')) 

model.compile(loss ='binary_crossentropy', 
					optimizer ='rmsprop', 
				metrics =['accuracy']) 

train_datagen = ImageDataGenerator( 
				rescale = 1. / 255, 
				shear_range = 0.2, 
				zoom_range = 0.2, 
			horizontal_flip = True) 

test_datagen = ImageDataGenerator(rescale = 1. / 255) 

train_generator = train_datagen.flow_from_directory(train_data_dir, 
							target_size =(img_width, img_height), 
					batch_size = batch_size, class_mode ='binary') 

validation_generator = test_datagen.flow_from_directory( 
									validation_data_dir, 
				target_size =(img_width, img_height), 
		batch_size = batch_size, class_mode ='binary') 
print(train_generator.class_indices)
print("hello da")
model.fit_generator(train_generator, 
	steps_per_epoch = nb_train_samples // batch_size, 
	epochs = epochs, validation_data = validation_generator, 
	validation_steps = nb_validation_samples // batch_size) 

model.save_weights('load_100.h5') 

#model.load_weights('model_saved.h5')
#print("abcd")
#test_image= image.load_img("a.jpg", target_size = (img_width, img_height)) 
#test_image = image.img_to_array(test_image)
#test_image = numpy.expand_dims(test_image, axis = 1)
#test_image = test_image.reshape(img_width, img_height)
#print("a1")
#result = model.predict_classes(test_image)   
#print(result)
