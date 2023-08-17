#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import shutil


# In[2]:

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
#limit gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# In[3]:


data_dir = 'data'


# In[4]:


# allowed extensions
image_exts = ['jpeg', 'jpg', 'bmp', 'png']


# In[5]:


# view image and convert from BGR to RGB
# creates numpy array for the image
# img = cv2.imread(os.path.join('data', 'happy', 'smile.woman_.jpg'))
# img.shape
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


# In[6]:


# access both directories for happy and sad
for image_class in os.listdir(data_dir):
    # access each image in the respective directories
    for image in os.listdir(os.path.join(data_dir, image_class)):
        # get the image path 
        image_path = os.path.join(data_dir, image_class, image)
        try:
            #img = cv2.imread(image_path)
            # get extension of image
            tip = imghdr.what(image_path)
            # if the extension is not in the list of valid extension then remove it
            if tip not in image_exts:
                print('Image not in ext list {}'.formate(image_path))
                # remove the image path
                os.remove(image_path)
        except Exception as e:
            print("Issue with image {}".format(image_path))


# In[7]:


# loads images from directory 
# each sub directory is a class label
data = tf.keras.utils.image_dataset_from_directory(data_dir)

happy_dir = os.path.join(data_dir, 'happy')
sad_dir = os.path.join(data_dir, 'sad')

num_happy = len(os.listdir(happy_dir))
num_sad = len(os.listdir(sad_dir))

print(f"Number of images in 'happy' class: {num_happy}")
print(f"Number of images in 'sad' class: {num_sad}")


# In[8]:


# this converts the data from tensors to numpy arrays
data_iterator = data.as_numpy_iterator()


# In[9]:


# Images represented as numpy arrays
batch = data_iterator.next()


# In[10]:


# 1 equals sad
# 0 equals happy


# In[11]:


# number of images to produce
number_of_images = 6
# create a set of subplots 'ax' and a figure 'fig'
fig, ax = plt.subplots(ncols=number_of_images, figsize=(20,20))
# take only the first 'number_of_images' from batch
# get the index and the value using enumerate
for idx, img in enumerate(batch[0][:number_of_images]):
    # display the images
    ax[idx].imshow(img.astype(int))
    #display the labels
    ax[idx].title.set_text(batch[1][idx])


# In[12]:


data = data.map(lambda x, y: (x/255, y))


# In[13]:


data_iterator = data.as_numpy_iterator()


# In[14]:


batch = data_iterator.next()


# In[15]:


# number of images to produce
number_of_images = 6
# create a set of subplots 'ax' and a figure 'fig'
fig, ax = plt.subplots(ncols=number_of_images, figsize=(20,20))
# take only the first 'number_of_images' from batch
# get the index and the value using enumerate
for idx, img in enumerate(batch[0][:number_of_images]):
    # display the images
    ax[idx].imshow(img)
    #display the labels
    ax[idx].title.set_text(batch[1][idx])


# In[16]:


train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1) + 1


# In[17]:


train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


# In[18]:


model = Sequential()


# In[19]:


model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[20]:


model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])


# In[21]:


model.summary()


# In[22]:


logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


# In[23]:


hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])


# In[24]:


fig = plt.figure()
plt.plot(hist.history['loss'], color='blue', label='loss')
plt.plot(hist.history['val_loss'], color='yellow', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# In[25]:


fig = plt.figure()
plt.plot(hist.history['accuracy'], color='blue', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='yellow', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc='upper left')
plt.show()


# In[26]:


from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# In[27]:


precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()


# In[28]:


for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    precision.update_state(y, yhat)
    recall.update_state(y, yhat)
    accuracy.update_state(y, yhat)


# In[29]:


print(f'Precision:{precision.result().numpy()}')
print(f'Recall:{recall.result().numpy()}')
print(f'Accuracy:{accuracy.result().numpy()}')


# In[30]:


test_src = 'testimages/sadmanissupersadman.jpg'
img = cv2.imread(test_src)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# In[31]:


resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()


# In[32]:


np.expand_dims(resize, 0).shape


# In[33]:


yhat = model.predict(np.expand_dims(resize/255, 0))


# In[34]:


destination_path = ""
print(yhat)
if(yhat > .5):
    print("Predicted Sad")
else:
    print("Predicted Happy")

answer = input("Correct Answer: ")
print(answer)
try:
    if(answer == "Happy" or answer == "happy"):
        source_path = test_src
        destination_path = 'data/happy/'
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} to {destination_path}")
    elif(answer == "Sad" or answer == "sad"):
        source_path = test_src
        destination_path = 'data/sad/'
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} to {destination_path}")
except Exception as e:
    print("Issue with path {}".format(e))


# In[35]:


from tensorflow.keras.models import load_model


# In[36]:


model.save(os.path.join('models', 'happysadmodel.h5'))


# In[37]:


new_model = load_model(os.path.join('models', 'happysadmodel.h5'))


# In[38]:


new_model.predict(np.expand_dims(resize/255, 0))


# In[ ]:




