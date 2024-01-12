from tensorflow import keras
from tensorflow.keras.layers import (Input, LayerNormalization, Dense, Dropout)

def base_model(input_shape, classification, output_bias=None):
    """Make a Linear NN."""
    print('input_shape:',input_shape)

    model = keras.models.Sequential()

    model.add( Input(input_shape) )
    model.add( LayerNormalization() )

    if classification:
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        model.add( Dense(1, activation='sigmoid', bias_initializer=output_bias) )
    else:
        model.add( Dense(1) )
    model.add( Dropout(0.5) )

    return model
