import cv2
import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19

def Histogram(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def WatershedAlgorithm(img):
    gray = cv2.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
    plt.show()
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    plt.imshow(cv2.cvtColor(unknown, cv2.COLOR_BGR2RGB))
    plt.show()
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]
    return img



SIZE_X=SIZE_Y=224
BATCH_SIZE = 32
data=ImageDataGenerator(validation_split = 0.3,vertical_flip=True,
                        horizontal_flip=True,zoom_range=0.1)


training=data.flow_from_directory("G:\sammer\lung_colon_image_set\lung_image_sets",
                                  class_mode = "categorical",
                                  target_size = (SIZE_X,SIZE_Y),
                                  color_mode="rgb",
                                  batch_size = 32, 
                                  shuffle = False,
                                  subset='training',
                                  seed = 32)

validation=data.flow_from_directory("G:\sammer\lung_colon_image_set\lung_image_sets",
                                    class_mode = "categorical",
                                    target_size = (SIZE_X,SIZE_Y),
                                    color_mode="rgb",
                                    batch_size = 32, 
                                    shuffle = False,
                                    subset='validation',
                                    seed = 32)



resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in resnet50.layers:
	layer.trainable = False

model= resnet50.output
model=Flatten(name="Flatten")(model)
model=Dense(256,activation="relu")(model)
model=Dense(128,activation="relu")(model)
model= Dropout(0.2)(model)
out_layer=Dense(3,activation='softmax')(model)
model=Model(inputs=resnet50.input,outputs=out_layer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

resnet_history = model.fit(training,
                           batch_size=BATCH_SIZE,
                           epochs=20,
                           validation_data=(validation),
                           steps_per_epoch=200,
                           validation_steps=50,
                           verbose=1)

r=model.evaluate(validation)

#plotting accuracy of Resnet50
plt.plot(resnet_history.history['accuracy'])
plt.plot(resnet_history.history['val_accuracy'])
plt.title('model accuracy with ResNet')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0, 20)
plt.ylim(0,1)
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(resnet_history.history['loss'])
plt.plot(resnet_history.history['val_loss'])
plt.title('model loss with ResNet')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0, 20)
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in vgg19.layers:
	layer.trainable = False


model= vgg19.output
model=Flatten(name="Flatten")(model)
model=Dense(256,activation="relu")(model)
model=Dense(128,activation="relu")(model)
model= Dropout(0.2)(model)
out_layer=Dense(3,activation='softmax')(model)
model=Model(inputs=vgg19.input,outputs=out_layer)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

vgg_history = model.fit(training,
                           batch_size=BATCH_SIZE,
                           epochs=20,
                           validation_data=(validation),
                           steps_per_epoch=100,
                           validation_steps=50,
                           verbose=1)
q=model.evaluate(validation)

#plotting accuracy of vgg
plt.plot(vgg_history.history['accuracy'])
plt.plot(vgg_history.history['val_accuracy'])
plt.title('model accuracy with VGG19')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.xlim(0, 20)
plt.ylim(0,1)
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(vgg_history.history['loss'])
plt.plot(vgg_history.history['val_loss'])
plt.title('model loss with VGG19')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.xlim(0, 20)

plt.legend(['train', 'validation'], loc='upper left')
plt.show()
