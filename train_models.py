import json
import pdb
import argparse

import importlib
import inspect
# from sklearn.metrics import mean_squared_error as mse, r2_score
import pickle
import utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow_addons.metrics import F1Score
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tensorflow_addons.optimizers import COCOB
import uuid
import time
from keras.models import load_model
################################################################################
# save printouts to log file
import logging as logging
logFormatter = logging.Formatter("%(asctime)s %(filename)s:%(lineno)d [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

def init_tf(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # see https://www.tensorflow.org/guide/gpu
    gpus = tf.config.list_physical_devices('GPU')
    logging.info(gpus)
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        logging.info(f'{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs')
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logging.error(e)
    else:
        logging.warning('no GPUS!!')

def get_parameters(func):
    """
    https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names#218709
    """
    keys = func.__code__.co_varnames[:func.__code__.co_argcount][::-1]
    sorter = {j: i for i, j in enumerate(keys[::-1])} 
    values = func.__defaults__[::-1]
    kwargs = {i: j for i, j in zip(keys, values)}
    sorted_kwargs = {
        i: kwargs[i] for i in sorted(kwargs.keys(), key=sorter.get)
    }   
    return sorted_kwargs

from utils import initializer

class Trainer:
    @initializer
    def __init__(
        self,
        data_dir=None,
        data_prefix='MFM',
        sample_limit='all',
        sample_rate=0.25,
        horizon=0,
        lab_order_delay=30,
        missingness=0.3,
        window=60,
        savedir='results_test/',
        random_state=42,
        ml="CNN_multiscale_classification_maxpool",
        initial_model=None,
        gpu=0,
        epochs=100,
        batch_size=128,
        features=['toco', 'fecg'],
        threshold=7.15,
        split=0.8,
        fit_kwargs={},
        lr=1e-3,
        min_lr=1e-6,
        step_patience=30,
        criteria='val_AUROC',
        time=time.time(),
        weighted=False,
        scale=True,
        missing='ffill',
    ):
        self.run_id = str(uuid.uuid4())
        self.datetime = str(datetime.now()).replace(' ','_') + '.log'

        str_sample_rate = (f'{sample_rate:.2f}').replace('.','')
        self.trainfile=(
            f'{data_dir}/{data_prefix}_train.'+'.'.join([
                f'sample_limit-{sample_limit}',
                f'sample_rate-{str_sample_rate}',
                f'horizon-{horizon}',
                f'window-{window}',
                f'lab_order_delay-{lab_order_delay}',
                f'missingness-{int(missingness*100)}',
                'parquet'
            ])
        ) 
        if 'CTU' in data_prefix:
            self.trainfile = self.trainfile.replace(
                f'lab_order_delay-{lab_order_delay}.', 
                ''
            )
        self.testfile=self.trainfile.replace('train','test')
        assert os.path.exists(self.trainfile), f"cannot find {self.trainfile}"
        assert os.path.exists(self.testfile), f"cannot find {self.testfile}"

    def save(self):
        """Save parameters of run to a json file."""
        with open(self.savedir+f'/run_{self.run_id}.json','w') as of:
            payload = {k:v for k,v in vars(self).items() 
                       if any(isinstance(v,t) for t in [bool, int, float, str])
                      }
            print('payload:',json.dumps(payload, indent=2))
            json.dump(payload,of, indent=4)

    def run(self):
        """Train a model and save it. Estimate its out-of-sample performance."""

        # logging
        os.makedirs(self.savedir, exist_ok=True)
        logfile = f'{self.savedir}/run_{self.run_id}.log'
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        rootLogger.setLevel(logging.DEBUG)

        # choose gpu
        init_tf(self.gpu)

        # get algorithm
        logging.info(f'import from methods.{self.ml}')
        algorithm = importlib.__import__('methods.'+self.ml,
                                        globals(),
                                        locals(),
                                        ['*']
                                        )
        make_model = algorithm.make_model
        self.supervised = algorithm.supervised
        self.classification = algorithm.classification

        if self.classification:
            self.target = 'pH Cord <'+str(self.threshold)
        else:
            self.target = 'LabResultValue'

        # optional keyword arguments passed to train
        if 'train_kwargs' in dir(algorithm):
            train_kwargs = algorithm.train_kwargs
            # for k,v in train_kwargs.items():
            #     if k in kwargs: 
            #         logging.warn(f'overriding {k} value defined by {ml} '
            #             + f'({train_kwargs[k]}) to {v}')
            #     else:
            #         setattr(p, k, v)
        self.nice_target = self.target.replace(' ','')
        if self.initial_model is None:
            self.model_name = f'{self.ml}_run_{self.run_id}'
        else:
            self.model_name = f'run_{self.run_id}'


        # make directories
        for folder in ['models','figures']:
            os.makedirs(os.path.join(self.savedir,folder), exist_ok=True)
    
        # load data
        data = utils.load_data(
            self.trainfile, 
            self.supervised,
            self.classification,
            label=self.target,
            features=self.features,
            split=self.split,
            missing=self.missing,
            random_state=self.random_state
        )
        if(self.testfile):
            test_data = utils.load_data(
                self.testfile, 
                self.supervised,
                self.classification,
                label=self.target,
                features=self.features,
                missing=self.missing,
                split=None,
                scale=False,
                random_state=self.random_state
            )
        if self.split==None:
            if self.scale:
                (X_train, y_train, scalers) = data
            else:
                (X_train, y_train) = data
            X_val = X_train
            y_val = y_train
        else:
            logging.info(f'splitting data (training={self.split})')
            if self.scale:
                (X_train, X_val, y_train, y_val, scalers) = data
            else:
                (X_train, X_val, y_train, y_val) = data
        

        if self.initial_model is not None:
            model = load_model(self.initial_model)
            # TODO: add checks that loaded model matches the ml name
        else:
            if self.classification:
                output_bias=np.log(np.sum(y_train)/len(y_train))
                model = make_model(input_shape=X_train.shape[1:],
                                output_bias=output_bias)
            else:
                model = make_model(input_shape=X_train.shape[1:])

        model.summary()
        logging.info(f'model_name: {self.model_name}')
        keras.utils.plot_model(
            model, 
            show_shapes=True,
            to_file=f'{self.savedir}/figures/{self.model_name}.png'
        )
        
            
        y_ratio = (y_train.shape[0] - y_train.sum())/ y_train.sum()
        default_fit_kwargs = dict(
            callbacks = [
                        keras.callbacks.ModelCheckpoint(
                            self.savedir+'models/'+self.model_name,
                            save_best_only=True, 
                            save_weights_only=False,
                            monitor=self.criteria,
                            mode = 'max'
                        ),

                        keras.callbacks.EarlyStopping(
                            monitor=self.criteria, 
                            patience=self.step_patience, 
                            verbose=1,
                            restore_best_weights=True,
                            mode = 'max'
                        ),
                        keras.callbacks.CSVLogger(logfile.replace('.log','.csv'), 
                                                    append=True, 
                                                    separator=';')
                    ],
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data = (X_val,y_val),
            verbose=1,
            shuffle = False,
            class_weight = {0: 1., 1: y_ratio if self.weighted else 1}  
        )
        # update fit arguments with local settings from model
        fit_kwargs = default_fit_kwargs
        fit_kwargs.update(self.fit_kwargs)

        self.loss = "binary_crossentropy" if self.classification else "mse"
        if self.classification:
            # TODO: pass multi_label=True to AUC when using multiple thresholds
            self.metrics = [
                tf.keras.metrics.BinaryCrossentropy(), 
                # Accuracy is not calculated correctly
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.AUC(curve='ROC', name='AUROC'),
                tf.keras.metrics.AUC(curve='PR', name='AUPRC'),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
                    
        else:
            self.metrics = [
                'mse',
                'mae',
                tf.keras.metrics.MeanAbsolutePercentageError(),
                utils.R_squared
            ]

        model.compile(
            # optimizer=Adam(learning_rate=self.lr),
            optimizer = COCOB(),
            loss=self.loss,
            metrics=self.metrics
        )

        ########################################################################
        # train model

        history = model.fit(X_train, y_train, **fit_kwargs)

        # save history
        with open(self.savedir+self.model_name+'.history','wb') as of:
            pickle.dump(history.history, of)
        print(self.savedir)
        print(self.model_name)
        #save model weights
        model.save(self.savedir + 'models/' + self.model_name)

        ######################################################################## 
        # Evaluate model on validation holdout
        val_metrics = model.evaluate(X_val, y_val, 
                                    batch_size=self.batch_size, 
                                    return_dict=True)
        for m, v in val_metrics.items():
            setattr(self,'Validation ' + m,v)
        # Evaluate model on test holdout
        if(self.testfile):
            X_test,y_test = test_data
            if self.scale:
                for i,scaler in enumerate(scalers):
                    X_test[:,:,i] = scalers[i].transform(X_test[:,:,i])
        else:
            X_test,y_test = X_val,y_val

        test_metrics = model.evaluate(X_test, y_test, 
                                    batch_size=self.batch_size, 
                                    return_dict=True)
        for m, v in test_metrics.items():
            print(f'Test {m}:{v}')
            logging.info(f'Test {m}:{v}')
            setattr(self,'Test ' + m,v)
        # save predictions on test
        pred_test = model.predict(X_test)
        pred_df = pd.DataFrame({
            'predicted prob':pred_test.ravel(),
            'label': y_test.ravel()
        })
        pred_df.to_csv(f'{self.savedir}models/{self.model_name}_predictions.csv',index = False)

        ############################################################################### 
        # save results
        self.save()
        return 

import fire
if __name__ == '__main__':
    fire.Fire(Trainer)
