from .Transformer import base_model
classification=True
supervised=True

def make_model(input_shape,output_bias):
    return base_model(
    input_shape,
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128], # change the number of parameter and see if it can fit
    mlp_dropout=0.4,
    dropout=0.25,
    # https://keras.io/examples/timeseries/timeseries_transformer_classification/
)