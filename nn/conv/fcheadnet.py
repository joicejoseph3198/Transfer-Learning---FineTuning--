# import necessary packages 
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense

class FCHeadNet:
    
    @staticmethod
    
    def build(baseModel, classes, D):

    # This method requires three parameters: 
    # the baseModel (the body of the network), 
    # the total number of classes in our dataset, 
    # and finally D, the number of nodes in FC layer.
        
        
        # initialize the model that will be placed at the 
        # top of the base and then add a FC layer.
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel) 

        return headModel
