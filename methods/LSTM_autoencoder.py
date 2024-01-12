from tensorflow import keras
from keras.layers import (InputLayer, LayerNormalization,
                          LSTM, RepeatVector, TimeDistributed,
                          Dense)
supervised=False
classification=False

def make_model(input_shape):
    """input shape is (timesteps, features)"""
#     input_shape = (None, input_shape[0], input_shape[1])
    print('input_shape:',input_shape)
    
    model = keras.models.Sequential()
    model.add(InputLayer(input_shape))
    # model.add(LayerNormalization())
#     input_layer = Input(input_shape)
    #model.add(LSTM(100, activation='relu' ))
    model.add(LSTM(100, activation='tanh' ))
    model.add(RepeatVector(input_shape[0]))
    #model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    # model.compile(optimizer='adam', loss='mse')
#     model.build(input_shape=input_shape)
    return model
