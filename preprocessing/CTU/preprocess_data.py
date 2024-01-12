# Maternal Fetal Monitoring

# > - William La Cava, william.lacava@childrens.harvard.edu

# - Extract the CTU-UHB fetal monitor data files into contiguous time-series.
# - Simple cleaning of extreme values
# - Plotting of individual data series for inspection & data exploration. 
# debug=True
from pqdm.processes import pqdm
import pandas as pd
import numpy as np
import os
import math
from collections import Counter

import warnings
warnings.simplefilter("ignore")
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter("ignore", SettingWithCopyWarning)

from utils import smooth

def load_file(data_path, f):
    df = pd.read_csv(f'{data_path}/{f}', index_col='time')
    df = df.rename(columns={
        'FHR':'fecg',
        'UC':'toco',
    })
    # df = df.drop(columns='time')
    # df.index.name='wct'
    PID = df['PID'].values[0]
    return PID, df

def process_sample(data_path, f, horizon, window, sample_rate, missingness):
    """Processess a single patient's time-series data."""
    try:
        PID, s = load_file(data_path, f)
    except Exception:
        print(f,'file load failure')
        # traceback.print_exc()
        return str("file load failure")

    sr = 4. # raw sample rate
    # calculate raw sample window
    raw_sample_window = window*60*sr
    # print('raw sample window:',raw_sample_window)
    sample_stride =  math.floor(sr/sample_rate) # sample stride to hit sample rate
    # clean duplicate time spots
    s.index = pd.DatetimeIndex(
        pd.date_range(start='2023-12-20', periods=len(s), freq='250ms')
    )
    # limit to samples within horizon+window of last reading 
    s_trim = s[
        s.index > (s.index.max() - pd.Timedelta(minutes=horizon+window))
    ]

    # trim data after horizon
    s_trim = s_trim[
        s_trim.index < (s_trim.index.max() - pd.Timedelta(minutes=horizon))
    ]

    min_index = s_trim.index.max() - pd.Timedelta(minutes=window)
    if s_trim.index.min() != min_index:
        s_trim.loc[min_index, 'fecg'] = np.nan
        s_trim.loc[min_index, 'toco'] = np.nan
    # must have less than 30% missingness
    if(s_trim.shape[0] < (1-missingness)*raw_sample_window):
        return "Too much missingness"
    # s_trim = s_trim[-math.ceil(.7*raw_sample_window):]
    # fill nans in missing samples 
    s_trim = s_trim.resample(pd.Timedelta(1/sr,'S')).asfreq()
    assert s_trim.shape[0] == 14401
    # smooth data using window corresponding to sample_rate Hz
    s_trim = s_trim.apply(lambda x: smooth(x, sample_stride+1, series=True))

    # downsample data to sample_rate Hz
    s_down = (
        s_trim.resample(pd.Timedelta(1/sample_rate,'S'), label='right')
        .apply(np.nanmean)
    )
    # need to check less than 30% missingness again, since down-sampling 
    # sometimes increases missingness a bit
    if any(s_down.isna().sum()/len(s_down) > missingness):
        return "Too much missingness"
    ts = dict(
        PID=PID,
        fecg = s_down['fecg'].values,
        toco = s_down['toco'].values,
        last_time = s_down.index.max()
    )
    # save
    return ts

def process_failures(seqs):
    """Prints stats on why loading strips failed and returns only successes"""
    ts_pass = [ts for ts in seqs if isinstance (ts, dict)]
    ts_fails = [ts for ts in seqs if isinstance (ts, str)]
    print(f'{len(ts_pass)} passes out of {len(seqs)}')
    print(f'{len(ts_fails)} failures out of {len(seqs)}')
    print(f'{len(seqs) - len(ts_pass) - len(ts_fails)} unexplained')
    cn = Counter(ts_fails)
    for k,v in cn.items():
        print(f"'{k}' removed {v} entries ({v/len(seqs)*100:.2f} %)")
    return ts_pass


def clean_lab(labvalue):
    """clean lab value"""
    labvalue = str(labvalue).replace('\\r','').replace('\\n','').replace("'",'')
    try:
        value = float(labvalue)
    except:
        if labvalue.startswith('<'):
            value = float(labvalue[1:])-0.1
        else:
            return float('nan')
    return value 

def run(
    sample_limit=1000,
    sample_rate=0.25,
    data_dir= '../../data/ctu-data/',
    save_dir='../../data/ctu-data/',
    n_jobs=10,
    horizon=0,
    window=60,
    # lab_order_delay=30,
    # warm=False,
    debug=False,
    seed=42,
    missingness=0.3,
    split=0.1812 #to get 100 train samples
):

    args=locals()
    for k,v in args.items():
        print(k,'=',v)

    os.makedirs(save_dir, exist_ok=True)
    np.random.seed(seed)
    ### load data path
    data_path = data_dir

    ending='.csv'

    fnames = [f for f in os.listdir(data_path) if f.endswith(ending)]

    print(f'found {len(fnames)} files')

    ########################################
    # turn arguments into filename
    argnames = []
    if sample_limit < len(fnames):
        argnames.append(f'sample_limit-{sample_limit}')
    else:
        argnames.append('sample_limit-all')
    str_sample_rate = (f'{sample_rate:.2f}').replace('.','')
    argnames.append(f'sample_rate-{str_sample_rate}')
    for setting in ['horizon','window']:
        argnames.append(f'{setting}-{args[setting]}')
    argnames.append(f'missingness-{int(missingness*100)}')

    # warmargnames = ".".join([a for a in argnames if 'lab_order_delay' not in a])
    # warmfilename = f'{save_dir}/processed-strips.{warmargnames}.parquet'
    data_name = save_dir + '/'+'.'.join(['MFM_CTU_all'] + argnames)
    train_name = save_dir + '/'+'.'.join(['MFM_CTU_train'] + argnames)
    test_name = save_dir + '/'+'.'.join(['MFM_CTU_test'] + argnames)
    print('training filename:',data_name)
    # print('testing filename:',test_name)


    ################################################################################
    # load and filter samples

    # - filter rules:
    #     - last contiguous section is at least _minimum sequence length_
    #     - equal toco - fecg count?
    #     - truncate the last contiguous sequence _minimum sequence length_?

    meta_list, ts_list = [], []
    corrupt = []
    i = 0

    ###############################
    # downsample signals to sample_rate Hz
    ###############################


    # match patient IDs with clinical data, specifically the target, PHCord.
    # labs = pd.read_csv(data_path+'labels.csv', error_bad_lines=False)
    # labs.loc[:,'PID'] = labs.loc[:,'filename'].apply(lambda x: str(x[:-4]))
    # PHcord = labs.loc[labs.LabTestName=='pH Cord']
    # init_ph_len = len(PHcord)
    # PH_PIDs = [str(p) for p in PHcord['PID'].values]
    # PHcord = PHcord.drop_duplicates(
    #                             subset = ['PID'],
    #                             # keep = 'first'
    #                             keep = False
    #                         )
    # get phCHord from labels.csv
    df_labels = pd.read_csv(data_path+'labels.csv', index_col='labname', error_bad_lines=False)
    # import pdb
    # pdb.set_trace()
    frames = []
    for col in df_labels.columns:
        frames.append({
            'PID':col,
            'LabResultValue':df_labels.loc['pH',col]
        })
    PHcord = pd.DataFrame(frames) 
    PHcord['LabResultValueFloat'] = PHcord['LabResultValue'].astype(float)
    PHcord['PID'] = PHcord['PID'].astype(int)
    # PHcord
    print(f'there are {len(PHcord)} PHcord measures. ')
    ###############


    ############################################################################
    # load data
    # if warm:
    #     print(f'reading ts_df from {warmfilename}...')
    #     ts_df = pd.read_parquet(warmfilename)
    # else:
    if debug:
        ts_list = []
        for fname in fnames:
            ts_list.append(
                process_sample(
                    data_dir, 
                    fname, 
                    horizon, 
                    window, 
                    sample_rate,
                    missingness
                    )
            )
    else:
        if sample_limit < len(fnames):
            # do in batches
            ts_list = []
            i = 0
            step = int(sample_limit/2)
            iters = 0
            while len(ts_list) < sample_limit and len(fnames) > i:
                # process in parallel
                # import ipdb
                # ipdb.set_trace()
                args = [
                    (data_dir, f, horizon, window, sample_rate, missingness)
                    for f in fnames[i:i+step]
                ]
                ts_batch = pqdm(
                    args, 
                    process_sample, 
                    argument_type='args',
                    n_jobs = n_jobs
                )
                ts_pass = process_failures(ts_batch)
                i += step
                ts_list += ts_pass.copy()
                iters += 1
            ts_list = ts_list[:sample_limit]

        else:
            # process in parallel
            args = [
                (data_dir, f, horizon, window, sample_rate, missingness)
                for f in fnames
                ]
            ts_list = pqdm(
                args, 
                process_sample, 
                n_jobs=n_jobs,
                argument_type='args',
            )
    ts_list = process_failures(ts_list)
    ############################################################################
    # save data
    ## make dataframes
    ts_df = pd.DataFrame.from_records(ts_list)
    

    print('total samples after processing time series:',len(ts_df))
    ############################################################################
    ## add PHcord label
    # ipdb.set_trace()
    ts_df = ts_df.merge(PHcord[[
        'PID',
        'LabResultValue',
        'LabResultValueFloat',
        # 'LabOrderDtime'
        ]], 
        on='PID',
        how='left'
    )
    print('total samples after merging labels:',len(ts_df))

    ################################################################################
    # make cutoffs
    cutoffs = [7.05, 7.1, 7.15, 7.2]
    for cutoff in cutoffs:
        target = 'pH Cord <'+str(cutoff)
        ts_df.loc[:,target] = ts_df['LabResultValue'].apply(lambda x: x < cutoff) 
        # # remove nan labels
        # ts_df = ts_df.loc[~ts_df[target].isna(),:]
        tmp = ts_df.loc[~ts_df['LabResultValue'].isna(),:]
        print(target+' incidence: {}/{} ({:.2f}%)'.format(
            tmp[target].sum(),
            len(tmp),
            tmp[target].sum()/len(tmp)*100)
        )
    ################################################################################
    # split data into train and test
    # need to split labelled and unlabelled data separately, then combine
    PIDs = ts_df.PID.unique()
    labelled_PIDs = ts_df.loc[~ts_df['LabResultValue'].isna(), 'PID'].unique()
    unlabelled_PIDs = ts_df.loc[ts_df['LabResultValue'].isna(), 'PID'].unique()
    print('total labelled samples:',len(labelled_PIDs))
    print('total unlabelled samples:',len(unlabelled_PIDs))
        
    labelled_train_samples = np.random.choice(labelled_PIDs, 
                                    size=int(len(labelled_PIDs)*split), 
                                    replace=False)

    unlabelled_train_samples = np.random.choice(unlabelled_PIDs, 
                                    size=int(len(unlabelled_PIDs)*split), 
                                    replace=False)
    train_samples = np.concatenate((labelled_train_samples, unlabelled_train_samples))

    ts_train = ts_df.loc[ts_df.PID.isin(train_samples),:]
    ts_test = ts_df.loc[~ts_df.PID.isin(train_samples),:]

    print('total samples:',len(ts_df))
    print('total testing samples:',len(ts_test))
    print('saving training and test files')

    ############################################################################
    # save parquet
    print(data_name+'.parquet','...')
    ts_df.reset_index().to_parquet(data_name+'.parquet')
    print(train_name+'.parquet','...')
    ts_train.reset_index().to_parquet(train_name+'.parquet')
    print(test_name+'.parquet','...')
    ts_test.reset_index().to_parquet(test_name+'.parquet')

import fire
if __name__ == '__main__':
    fire.Fire(run)