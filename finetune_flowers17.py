# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from datasets.simpledatasetloader import SimpleDatasetLoader
from nn.conv.fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to output model")
args = vars(ap.parse_args())

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
shear_range=0.2, height_shift_range=0.1, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

# grab the list of images and extract labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [img.split(os.path.sep)[-2] for img in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# Again, we make the assumption that our input dataset has the following
# directory structure: dataset_name/{class_name}/example.jpg

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224,224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk and scale it 
sdl = SimpleDatasetLoader(preprocessors=[aap,iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0 

# partition the data 
(trainX, testX, trainY, testY) = train_test_split(data, labels,
test_size=0.25, random_state=42)

# converting the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY  = LabelBinarizer().fit_transform(testY)


# load the VGG16 network
baseModel =  VGG16(weights="imagenet", include_top=False,
input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs = baseModel.input, outputs = headModel)

# freeze the weights in the body so they are not updated 
# during the backpropagation phase
for layer in baseModel.layers:
    layer.trainable = False

# compile the model 
print("[INFO] compiling the model...")
opt = RMSprop(lr = 0.001)
model.compile(loss = "categorical_crossentropy", optimizer=opt,
metrics = ["accuracy"])

# allow the new FC layers to start to become initialized with 
# actual "learned" values for a few epochs
print("[INFO] training head ...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
validation_data = (testX, testY), epochs=25, 
steps_per_epoch = len(trainX)//32, verbose =1)

# evalutate network after initialization
print("[INFO] evaluating after initialization")
predictions = model.predict(testX, batch_size =32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),target_names=classNames))

# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True
    # If classification accuracy continues to improve 
    # (without overfitting), you may want to consider 
    # unfreezing more layers in the body.

# for the changes to the model to take affect we need to 
# recompile the model
print("[INFO] re-compiling the model ... ")
opt = SGD(lr =0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics = ["accuracy"])

# train the model again, this time fine-tuning *both* 
# the final set of CONV layers along with our set of FC layers
print("[INFO] fine-tuning model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
validation_data=(testX, testY), epochs=100, 
steps_per_epoch=len(trainX) //32, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])