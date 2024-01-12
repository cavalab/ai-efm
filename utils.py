import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import ipdb
import tensorflow as tf
from tqdm import tqdm
import logging 
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def R_squared(y, y_pred):
  residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
  total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
  r2 = tf.subtract(1.0, tf.divide(residual, total))
  return r2

# plotting tools
def plot_data(data, sample_rate_hz=4, normalize=False, h=None, label=None):
    import matplotlib.pyplot as plt
    if h == None:
        h = plt.figure(figsize=(30,6))
        ax0 = h.add_subplot(2,2,1) 
        ax1 = h.add_subplot(2,2,3) 
    else:
        axes = h.get_axes()
        ax0 = axes[0] 
        ax1 = axes[1] 
#     ax3 = h.add_subplot(2,2,(2,4))
    max_mins = len(data[0])/sample_rate_hz/60
    minutes = np.linspace(0,max_mins,len(data[0]))
    for x in data:
        if normalize:
            x = (x-np.mean(x, axis=0))/np.std(x, axis=0)
        ax0.plot(minutes, x[:,0], label=label) #, alpha=5/len(data))
        ax1.plot(minutes, x[:,1], label=label) #, alpha=5/len(data))
    ax0.set_ylabel('Tocogram')
    ax1.set_ylabel('Heart Rate')
    plt.xlabel('Minutes')
#     h2 = plt.figure()
#     for x in data:
#         ax3.scatter(x[::100,0],x[::100,1])
################################################################################
import numpy as np
import pandas as pd
import logging as log

def load_parquet(data_path, supervised, classification,
                 label, features, split, missing_data_method):
    """load a parquet data file"""

    log.info(f'reading parquet file: {data_path}')
    df = pd.read_parquet(data_path)

    if supervised:
        # keep only labelled samples
        df = df.loc[~df['LabResultValueFloat'].isna(),:]
    else:
        # keep only unlabelled samples
        df = df.loc[df['LabResultValueFloat'].isna(),:]
    
    X = np.stack([np.vstack(df[f]) for f in features], axis=2)
    
    # Create a new array to store the filled data
    X_filled = np.empty_like(X)

    # Handle missing data based on missing_data_method
    for i in range(X.shape[2]):
        if missing_data_method == 'ffill':
            # Fill missing data with the previous value, then the next value if still missing
            X_filled[:, :, i] = np.apply_along_axis(
                lambda x: (pd.Series(x)
                    .fillna(method='ffill')
                    .fillna(method='bfill')
                    .values
                    ), 
                axis=1, 
                arr=X[:, :, i]
            )
        elif missing_data_method == 'zeros':
            # Fill missing data with 0
            X_filled[:, :, i] = X[:, :, i].copy()
            X_filled[:, :, i][np.isnan(X[:, :, i])] = 0
        elif missing_data_method == 'drop':
            # Drop all NaNs, concatenate the rest, and take the first 70% of data
            max_missingness = np.isnan(X[:,:,i]).sum(axis = 1).max()
            cutoff = X.shape[1] - max_missingness

            for j in range(X.shape[0]):
                x = pd.Series(X[j, :, i]).dropna().values
                X_filled[j, :cutoff, i] = x[:cutoff]

            # Resize X_filled to the new shape
            X_filled = X_filled[:, :cutoff, :]

    log.info(f'samples: {X_filled.shape[0]},'
             +f'time series length: {X_filled.shape[1]},'
             +f'features: {X_filled.shape[2]}'
          )
    

    y = df[label].values if supervised else X_filled

    return X_filled, y



from dask import dataframe as dd
def load_json(data_path, supervised, classification,
              label, features, split):
    """load json data"""
    log.info('reading file...')

    df = dd.read_json(data_path, 
                      blocksize=2**30
                      ) #.set_index('PID')

    if supervised:
        # keep only labelled samples
        df = df.dropna(subset=[label])
    else:
        # keep only unlabelled samples
        df = df.loc[df['LabResultValueFloat'].isna(),:]
    
    X = np.stack([np.vstack(df[f].compute()) for f in features], axis=2) 
    log.info(f'samples: {X.shape[0]},'
             +f'time series length: {X.shape[1]},'
             +f'features: {X.shape[2]}'
          )
    # ipdb.set_trace()

    assert np.all(X[0,:,0] == df[features[0]].compute().values[0])
    y = df[label].values.compute() if supervised else X
    return X,y

#from tf.data import TFRecordDataset
import copy
def load_data(data_path, supervised=True, classification=True,
              label='pH Cord <7.15', features=['toco','fecg'],
              split=None, missing = 'ffill', scale=True,
              random_state=42
              ):
    """load a data file and return train/test splits."""

    if data_path.endswith('.parquet'):
        X,y = load_parquet(data_path,supervised,classification,label,
                            features, split,missing)
    elif data_path.endswith('.json'):
        X,y = load_json(data_path,supervised,classification,label,
                            features, split)
    else:
        ValueError(f'data_path invalid: {data_path}')

    # make y an int
    y = y.astype(np.int32)

    log.info(f'X shape: {X.shape}')
    if classification:
        log.info(f'y has {np.sum(y)} cases and {np.sum(y==0)} controls')

    if scale:
        scalers = []
        for i, f in enumerate(features):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            X[:,:,i] = scaler.fit_transform(X[:,:,i])
            scalers.append(copy.deepcopy(scaler))

    if split == None:
        if scale:
            return X,y,scalers  
        else:
            return X,y
    else:
        if supervised:
            # log.info('label:',label)
            stratify=y if classification else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y,
                stratify=stratify,
                random_state=random_state,
                train_size=split
            )
            log.info(f'{np.mean(X_train[:,0])} training fecg mean and {np.mean(X_test[:,0])} testing fecg mean')
            log.info(f'{len(y_train)} training and {len(y_test)} testing labelled'
                   ' samples')
        else:
            X_train, X_test = train_test_split(
                X, 
                random_state=random_state, 
                train_size=split
            )
            y_train = X_train
            y_test = X_test

        if scale:
            return X_train, X_test, y_train, y_test, scalers  
        else:
            return X_train, X_test, y_train, y_test  


from functools import wraps
import inspect

def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    # names, varargs, keywords, defaults = inspect.getargspec(func)
    names, varargs, keywords, defaults, _,_,_ = inspect.getfullargspec(func)
    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper