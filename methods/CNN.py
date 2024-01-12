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
    """Make a CNN regressor."""
    print('input_shape:',input_shape)
    conv_params = dict(filters=100, 
                       kernel_size=10,
                       padding="same",
                       activation="relu"
                      )
    pool_params = dict(padding="same", pool_size=2, strides=2)

    model = keras.models.Sequential()

    model.add( Input(input_shape) )
    model.add( LayerNormalization() )

    model.add( Conv1D(**conv_params) )
    model.add( BatchNormalization() )
    # model.add( MaxPooling1D(**pool_params) )
    model.add( AveragePooling1D(**pool_params) )
    #model.add( Dropout(0.5) )

    model.add( Conv1D(**conv_params) )
    model.add( BatchNormalization() )
    # model.add( MaxPooling1D(**pool_params) )
    model.add( AveragePooling1D(**pool_params) )
    #model.add( Dropout(0.5) )

    model.add( Conv1D(**conv_params) )
    model.add( BatchNormalization() )
    # model.add( MaxPooling1D(**pool_params) )
    model.add( AveragePooling1D(**pool_params) )
    #model.add( Dropout(0.5) )

    model.add( Conv1D(**conv_params) )
    model.add( BatchNormalization() )
    # model.add( MaxPooling1D(**pool_params) )
    model.add( AveragePooling1D(**pool_params) )
    #model.add( Dropout(0.5) )

    model.add( Conv1D(**conv_params) )
    model.add( BatchNormalization() )
    model.add( MaxPooling1D(**pool_params) )
    # model.add( AveragePooling1D(**pool_params) )
    #model.add( Dropout(0.5) )

    model.add( GlobalAveragePooling1D() )
    model.add( Dropout(0.25) )

    if classification:
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    else:
        model.add( Dense(1) )

    # return keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model
