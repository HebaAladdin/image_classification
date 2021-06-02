from __future__ import absolute_import, division

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras import regularizers

# method to return the dense classifier
def get_classifier(input_length,layer_sizes=[100],output_size=2):
    
    model = Sequential()
    model.add(Flatten(input_shape=[input_length]))    
    for layer_size in layer_sizes:
        if layer_size < 1:
            model.add(Dropout(layer_size)) 
        else:
            model.add(Dense(layer_size, activation='relu'))
   
      
    model.add(Dense(output_size, activation='softmax', name="predictions"))
    
    return model