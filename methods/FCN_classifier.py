from .FCN import base_model
supervised=True
classification=True

def make_model(input_shape, output_bias=None):
    return base_model(input_shape, classification, output_bias)
