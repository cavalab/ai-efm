from .Linear import base_model
supervised=True
classification=False

def make_model(input_shape):
    return base_model(input_shape, classification)
