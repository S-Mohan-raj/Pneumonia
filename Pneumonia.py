# Import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.optimizers import Adam
import cv2
from keras.utils import img_to_array

# Creating training data generator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

# Train the data which is in preset drive
train_images="/content/chest_xray/train"
train_generator= train_datagen.flow_from_directory(train_images,
                                                   target_size=(300,300),
                                                   batch_size=128,
                                                   class_mode='binary')

# validation generator and loading validation data
test_datagen=ImageDataGenerator(rescale=1./255)
validation_generator=test_datagen.flow_from_directory('content/chest_xray/val',
                                                      target_size=(300,300),
                                                      batch_size=128,
                                                      class_mode='binary')

# Plotting the images without pneumonia
plot_image=plt.figure(figsize=(10,10))
plot1 = plot_image.add_subplot(3,2,1)
plot2 = plot_image.add_subplot(3,2,2)
plot3 = plot_image.add_subplot(3,2,3)
plot4 = plot_image.add_subplot(3,2,4)
plot5 = plot_image.add_subplot(3,2,5)
plot6 = plot_image.add_subplot(3,2,6)
plot1.matshow(plt.imread(train_generator.filepaths[41]))
plot2.matshow(plt.imread(train_generator.filepaths[176]))
plot3.matshow(plt.imread(train_generator.filepaths[1553]))
plot4.matshow(plt.imread(train_generator.filepaths[354]))
plot5.matshow(plt.imread(train_generator.filepaths[2679]))
plot6.matshow(plt.imread(train_generator.filepaths[2710]))

# with pneumonia
plot_image=plt.figure(figsize=(10,10))
plot1 = plot_image.add_subplot(3,2,1)
plot2 = plot_image.add_subplot(3,2,2)
plot3 = plot_image.add_subplot(3,2,3)
plot4 = plot_image.add_subplot(3,2,4)
plot5 = plot_image.add_subplot(3,2,5)
plot6 = plot_image.add_subplot(3,2,6)
plot1.matshow(plt.imread(train_generator.filepaths[1419]))
plot2.matshow(plt.imread(train_generator.filepaths[1365]))
plot3.matshow(plt.imread(train_generator.filepaths[1400]))
plot4.matshow(plt.imread(train_generator.filepaths[1350]))
plot5.matshow(plt.imread(train_generator.filepaths[1345]))
plot6.matshow(plt.imread(train_generator.filepaths[1349]))

# Neural Network Using Tensorflow
model=tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),])

model.summary()
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

# Training the model
history=model.fit(train_generator,epochs=20,validation_data=validation_generator)

# Validation loss
loss=history.history['loss']
val_loss=history.history['val_loss']

# Plotting loss vs no.of epochs
plt.figure(figsize=(15,10))
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['Traning loss','Validation loss'],fontsize=16)
plt.title(" Loss vs Epochs ",fontsize=18)
plt.xlabel(" Number of Epochs ",fontsize=16)
plt.ylabel(" Loss ",fontsize=16)
print(" Loss vs Number of Epochs ")
plt.show()

# Validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Plotting accuracy vs no.of epochs
plt.figure(figsize=(15,10))
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['Traning accuracy','Validation accuracy'],fontsize=16)
plt.title(" Accuracy vs Epochs ",fontsize=18)
plt.xlabel(" Number of Epochs ",fontsize=16)
plt.ylabel(" Accuracy ",fontsize=16)
print(" Accuracy vs Epochs ")
plt.show()

# Save the model
model.save(" trained.h5 ")

# Load the model
eval_datagen=ImageDataGenerator(rescale=1/255)
test_generator=eval_datagen.flow_from_directory('/content/chest_xray/test',
    target_size=(300,300),
    batch_size=128,
    class_mode='binary')

eval_result=model.evaluate_generator(test_generator,624)
print('loss : ',eval_result[0])
print('accuracy : ',eval_result[1])

# Predictions
img=cv2.imread('/content/chest_xray/test/NORMAL/IM-0003-0001.jpeg')
tempimg=img
img=cv2.resize(img,(300,300))
img=img/255.0
img=img.reshape(1,300,300,3)
model.predict(img)
prediction=model.predict(img)>=0.5
if prediction>=0.5:
    prediction=" Pnemonia "
else:
    prediction=" Normal "
print(" Prediction = "+prediction)
plt.imshow(tempimg)
plt.title(" Prediction = "+prediction,fontsize=14)