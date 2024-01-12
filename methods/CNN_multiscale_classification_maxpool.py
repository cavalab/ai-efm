from tensorflow import keras
from keras.layers import Dense 
from keras.models import Model
from .CNN_multiscale import make_model as make_ms_model
supervised=True
classification=True

def make_model(input_shape, output_bias=None, pooling = 'max'):
    """This is the main model.
    It takes the original time series and its down-sampled versions as an input, 
    and returns the result of classification as an output.
    """
    return  make_ms_model(input_shape, classification, pooling,output_bias)

