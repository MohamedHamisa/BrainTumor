import cv2 #opencv read rgb img only  
import os
from PIL import image #python image library provides python interpreter with image editing capability
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras,model import Seqeuntial
from keras.layers import Conv2D, MaxPooling2D,Activation,Droput,Flattten,Dense
from sklearn.model_selection import tran_test_split #to split data into train and test
from keras.utils import normalize #to normalize the data
#from keras.utils import to_categorical  #to work with more than 1 output#
image_directory = ' '  #you can get dataset from kaggle
#we are creating a list for the images in the folder that contains yes is tumor and no data  
no_tumor = os.listdir(image_directory + 'no/') #images in folder no
yes_tumor = os.listdir(image_directory + 'yes/') #images in folder yes
dataset=[] #to classify the images with 1 and 0
label=[]
input_size = 64
#dataset , label = 3000
#print(no_tumor)
#path = 'noo.jpg'
#print(path.split('.')[1]) #will split the extension
#enumertare gives images in data like keyvalues or keys to act like a counter  
for i , image_name in enumerate(no_tumor):
  if(image_name.split('.')[1]=='jpg'):
    image=cv2.imread(image_directory+'/no'+image_name)
    image = Image.fromarray(image,'RGB') #convert images to rgb format
    image = image.resize((64,64)) 
    dataset.append(np.array(image)) 
    label.append(0)

for i , image_name in enumerate(yes_tumor):
  if(image_name.split('.')[1]=='jpg'):
    image=cv2.imread(image_directory+'yes/'+image_name)
    image = Image.fromarray(image,'RGB') #convert images to rgb format
    image = image.resize((64,64)) 
    dataset.append(np.array(image)) 
    label.append(1)
dataset = np.array(dataset) #to convert data to array
label = np.array(label) 

x_train , y_trian , x_test , y_test = train_test_split(dataset,label,test_size=0.2,ranodm_state=0)
print(x_train.shape)
x_train = normalize(x_train,axis=1)
x_test = normalize(x_test,axis=1)
#Each element in the the axes that are kept is normalized independently
#If axis is set to 'None', the layer will perform scalar normalization (dividing the input by a single scalar value).
#The batch axis, 0, is always summed over (axis=0 is not allowed)
#For Dense layer, all RNN layers and most other types of layers, the default of axis=-1 is what you should use,
#For Convolution2D layers with dim_ordering=“th” (the default), use axis=1,
#For Convolution2D layers with dim_ordering=“tf”, use axis=-1 (i.e. the default).


#model define
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3,3),input_shape=(input_size,input_size,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Activation(tf.nn.relu))

model.add(tf.keras.layers.Conv2D(32, (3,3),kernal_initializer='hu_uniform')) #define the way to initialize the weights of keras layers randomly
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Activation(tf.nn.relu))

model.add(tf.keras.layers.Conv2D(32, (3,3),kernal_initializer='hu_uniform'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(tf.keras.layers.Activation(tf.nn.relu))


model.add(tf.keras.layers.Flatten(input_shape=())) #flat the input images to vectors
model.add(tf.keras.layers.Dense((64,activation=tf.nn.relu))
model.add(tf.keras.layers.Activation(tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)) #out
#sigmoid,Binarycrossentropy , 1 -->Binary
#softmax,Sparse,2 --> more than 1 
model.compile(optimizer='adam',
             loss='Binary_categorical_crossentropy',
             metrics=['accuracy']
             )

model.fit(x_train, y_train,batch_size=16,verbose=1 ,epochs=10,validation_data=(x_test,y_test),shuffle=False))

model.save('Brain Tumor.h5')




