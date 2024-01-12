import tensorflow as tf
from tensorflow import keras
from keras.layers import (Conv1D, Dense, Dropout, Input, Concatenate, 
                          GlobalMaxPooling1D, AveragePooling1D,MaxPooling1D)
from keras.models import Model

# from https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57

def get_base_model(input_shape, fsize):
    """this base model is one branch of the main model
    It takes a time series as an input, performs 1-D convolution, and returns it as an output ready for concatenation
    """
    print('get_base_model input_shape:',input_shape)
    # input_len = input_shape[0]
    # input_width = input_shape[1]
    #the input is a time series of length n and width input_width
    input_seq = Input(input_shape)
    #choose the number of convolution filters
    nb_filters = 100
    #1-D convolution and global max-pooling
    convolved = Conv1D(nb_filters,kernel_size=int(fsize), padding="same", activation="tanh",
            name='conv1d')(input_seq)
    processed = GlobalMaxPooling1D(name='maxpooling')(convolved)
    #dense layer with dropout regularization
    compressed = Dense(50, activation="tanh")(processed)
    compressed = Dropout(0.3)(compressed)
    model = Model(inputs=input_seq, outputs=compressed)
    return model

def make_model(input_shape, classification, pooling,output_bias=None):
    """This is the main model.
    It takes the original time series and its down-sampled versions as an input, 
    and returns the result of classification as an output.
    """
    print('input_shape:',input_shape)
    # fsizes = [8,16,24]
    ts_len = input_shape[0]
    fsizes = [int(ts_len/100), int(ts_len/20), int(ts_len/10)]
    print('filter sizes:',fsizes)
    inputs = Input(input_shape)
    #the inputs to the branches are the original time series, and its down-sampled versions
    if(pooling == 'max'):
        print('using max_pooling')
        input_smallseq = MaxPooling1D(pool_size=4,strides=4)(inputs)
        input_medseq = MaxPooling1D(pool_size=2,strides=2)(inputs)
        input_origseq = MaxPooling1D(pool_size=1,strides=1)(inputs)
    else:
        print('using avg_pooling')
        input_smallseq = AveragePooling1D(pool_size=4,strides=4)(inputs)
        input_medseq = AveragePooling1D(pool_size=2,strides=2)(inputs)
        input_origseq = AveragePooling1D(pool_size=1,strides=1)(inputs)
    # input_origseq = Input(input_shape)
    #the more down-sampled the time series, the shorter the corresponding filter
    width=input_shape[1]
    base_net_small = get_base_model(input_smallseq.shape[1:], fsizes[0])
    base_net_med = get_base_model(input_medseq.shape[1:], fsizes[1])
    base_net_original = get_base_model(input_shape, fsizes[2])

    embedding_small = base_net_small(input_smallseq)
    embedding_med = base_net_med(input_medseq)
    embedding_original = base_net_original(input_origseq) 
    # embedding_original._name = 'embedding_original'
    #concatenate all the outputs
    merged = Concatenate()([embedding_small, 
                            embedding_med, 
                            embedding_original])
    # inputs = [input_smallseq, input_medseq, input_origseq]
    # return inputs, merged
    if classification:
        if output_bias is not None:
            output_bias = keras.initializers.Constant(output_bias)
        out= Dense(1, activation='sigmoid', bias_initializer=output_bias)(merged)
    else:
        out= Dense(1)(merged)

    model = Model(inputs=inputs, #[input_smallseq, input_medseq, input_origseq], 
                  outputs=out)
    return model

