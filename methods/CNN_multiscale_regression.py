from keras.layers import Dense 
from keras.models import Model
from .CNN_multiscale import make_model as make_ms_model

supervised=True
classification=False

def make_model(input_shape):
    """This is the main model.
    It takes the original time series and its down-sampled versions as an input, 
    and returns the result of classification as an output.
    """
    return make_ms_model(input_shape, classification)

