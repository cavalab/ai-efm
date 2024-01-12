from tensorflow import keras
from tensorflow.keras.layers import (Input, 
                                     LayerNormalization, 
                                     Conv1D, 
                                     BatchNormalization, 
                                     GlobalAveragePooling1D, 
                                     MaxPooling1D, 
                                     AveragePooling1D,
                                     Dense, 
                                     Dropout)

def base_model(input_shape, classification, max = True, output_bias=None):
    """Make a fully connected CNN model.
    From: https://arxiv.org/abs/1611.06455v4
    """
    print('input_shape:',input_shape)
    conv_params = dict(
        padding="same",
        activation="relu"
    )
    pool_params = dict(padding="same", pool_size=2, strides=2)

    model = keras.models.Sequential()

    model.add( Input(input_shape) )
    model.add( LayerNormalization() )

    model.add( Conv1D(kernel_size=8, filters=128, **conv_params) )
    model.add( BatchNormalization() )

    model.add( Conv1D(kernel_size=5, filters=256, **conv_params) )
    model.add( BatchNormalization() )

    model.add( Conv1D(kernel_size=3, filters=128, **conv_params) )
    model.add( BatchNormalization() )


    model.add( GlobalAveragePooling1D() )

    if classification:
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    else:
        model.add( Dense(1) )

    return model
