from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing import image

from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Activation, Dropout, Flatten, Dense 
from keras import backend as K 
#import PIL as image
import numpy
img_height,img_width=224,224
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
				
model.load_weights('load_100.h5')

count=0
test_image= image.load_img("1.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1

test_image= image.load_img("2.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("3.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("4.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("5.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("6.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("7.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("8.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)  
if result == 0:
	count+=1
test_image= image.load_img("9.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("10.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image)
print(result) 
if result == 0:
	count+=1
test_image= image.load_img("11.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("12.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 0:
	count+=1
test_image= image.load_img("13.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 1:
	count+=1
	
test_image= image.load_img("14.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 1:
	count+=1
	
test_image= image.load_img("15.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 1:
	count+=1
	
test_image= image.load_img("16.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 1:
	count+=1
	
test_image= image.load_img("17.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image) 
print(result)
if result == 1:
	count+=1
	
test_image= image.load_img("18.jpg", target_size = (img_width, img_height)) 
test_image = image.img_to_array(test_image)
test_image = numpy.expand_dims(test_image, axis = 1)
test_image = test_image.reshape(-1,img_width, img_height,3)
result = model.predict_classes(test_image)
print(result)
if result == 1:
	count+=1 
	
print("Accuracy: ",round(count/18*100,2),"%")