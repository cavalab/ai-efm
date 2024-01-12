from tensorflow import keras
from tensorflow.keras.layers import (Input, 
                                     LayerNormalization, 
                                     Conv1D, 
                                     BatchNormalization, 
                                     GlobalAveragePooling1D, 
                                     MaxPooling1D, 
                                     AveragePooling1D,
                                     Dense, 
                                     Dropout,
                                     ReLU,
                                     add
                                     )

def add_block(model, shortcut, kernel_size, filters):

    model = Conv1D(kernel_size=kernel_size, filters=filters, activation=None)( model )
    model = BatchNormalization()( model )
    model = ReLU()( model )
    model = Conv1D(kernel_size=kernel_size, filters=filters, activation=None)( model )
    model = BatchNormalization()( model )
    model = ReLU()( model )
    model = Conv1D(kernel_size=kernel_size, filters=filters, activation=None)( model )
    model = BatchNormalization()( model )
    model = add([model, shortcut])
    model = ReLU()( model )
    new_shortcut = model
    return model, new_shortcut

def base_model(input_shape, classification, max = True, output_bias=None):
    """Make a residual network model.
    Architecture from: https://arxiv.org/abs/1611.06455v4
    """
    print('input_shape:',input_shape)
    conv_params = dict(
        padding="same",
        activation="relu"
    )
    pool_params = dict(padding="same", pool_size=2, strides=2)

    model = Input(input_shape)
    model = LayerNormalization()(model)
    # model.add( Input(input_shape) )
    # model.add( LayerNormalization() )
    shortcut = model

    model, shortcut = add_block( model, shortcut, 8, 64)
    model, shortcut = add_block( model, shortcut, 5, 128)
    model, _ = add_block( model, shortcut, 3, 128)

    model.add( GlobalAveragePooling1D() )

    if classification:
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    else:
        model.add( Dense(1) )

    return model
