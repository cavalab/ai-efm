from tensorflow import keras
from tensorflow.keras.layers import (InputLayer, LayerNormalization,
                          LSTM, BatchNormalization, Dense)
from tensorflow.keras.regularizers import l2

def base_model(input_shape, classification, output_bias=None):
    """input shape is (timesteps, features)"""
#     input_shape = (None, input_shape[0], input_shape[1])
    lstm_args = dict(units=24, 
                   activation='tanh',
                   # recurrent_activation='hard_sigmoid',
                   recurrent_activation='sigmoid',
                   kernel_regularizer=l2(3e-2),
                   recurrent_regularizer=l2(3e-2)
                   )
    print('input_shape:',input_shape)
    model = keras.models.Sequential()
    model.add(InputLayer(input_shape))
    model.add(LSTM(**lstm_args, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(**lstm_args, return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(**lstm_args, return_sequences=False))
    model.add(BatchNormalization())
    activation = 'sigmoid' if classification else None
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)
    model.add(Dense(1, activation=activation, 
                    bias_initializer=output_bias))

    return model
