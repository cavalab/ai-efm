# Maternal Fetal Monitoring

# > - Corey Chivers, corey.chivers@pennmedicine.upenn.edu
# > - William La Cava, lacava@upenn.edu

# - Extract the GE fetal monitor data files into contiguous time-series.
# - Simple cleaning of extreme values
# - Plotting of individual data series for inspection & data exploration. 
# debug=True
import ipdb
from pqdm.processes import pqdm
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import math
from collections import Counter
import copy

import warnings
warnings.simplefilter("ignore")
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter("ignore", SettingWithCopyWarning)
from tqdm import tqdm
import sys
import traceback

def smooth(x,window_len=11,window='hanning', series=False):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    # WGL: match input length
    y = y[round(window_len/2-1):-round(window_len/2)]
    # import ipdb
    # ipdb.set_trace()
    # return pd.Series(y, index=x.index) if series==True else y
    return pd.Series(y, index=x.index) if series==True else y


def read_file(data_path, fname):
    path = f'{data_path}{fname}'
    # print('loading',path)
    d = pd.read_csv(path, low_memory=False)
    d['Record Time'] = pd.to_datetime(d['Record Time'])
    d['Display Time'] = pd.to_datetime(d['Display Time'])
    return d


def gen_daterange(start, periods=240):
    return pd.Series(pd.date_range(start=start, freq='250ms', periods=periods))


def mkts(df, name):
    """Convert wide format into a time-series."""
    if df.shape[0] == 0:
        return None
    if any(str(i) not in df.columns for i in range(240)):
        return None
    # cleanup weird endlines
    tmp = df[[str(i) for i in range(240) if str(i) in df.columns]] 
    for col in tmp.columns:
        tmp.loc[:,col] = tmp[col].apply(lambda x : 
                                        float(str(x).replace('\\r','').replace('\\n','').replace("'",'')))
    
    # Sampling frequency is 4hz
    try:
        vals = tmp.values.reshape((240*tmp.shape[0],))
    except Exception as e:
        print('df:',df.shape,df.columns,df)
        print('tmp:',tmp.shape,tmp.columns,tmp)
        raise e
#     vals = df[[str(i) for i in range(240)]].values.reshape((240*df.shape[0],))
    times = pd.concat([gen_daterange(df['Record Time'].values[i]) for i in range(df.shape[0])])
    ts = (
        pd.DataFrame({name: vals, 'wct': times})
        .sort_values('wct')
        .set_index('wct')
        .replace(0, np.nan)
        .dropna()
    )
    return ts


def clean_ts(ts):
    """Simple outlier removal."""
    ts.loc[ts['toco'] >= 128, 'toco'] = np.nan
    ts.loc[ts['fecg'] < 30, 'fecg'] = np.nan
    ts.loc[ts['fecg'] > 220, 'fecg'] = np.nan
    ts.dropna(inplace=True)


def mkcontigs(comb, max_gap_hours=1):
    """Find start and end index tuples for contiguous spans of readings."""
    if comb.shape[0] ==  0:
        return [], []
    idb_breaks = ((comb.index.values[1:] - comb.index[:-1]) 
                    > np.timedelta64(max_gap_hours, 'h'))
    idb_breaks = np.append(idb_breaks,False)
    start_i = 0
    contigs_i = []
    contigs_t = []
    for i,v in enumerate(idb_breaks):
        if v:
            end_i = i
            contigs_i.append([start_i, end_i])
            contigs_t.append([comb.index.values[start_i], 
                              comb.index.values[end_i]])
            start_i = i+1
    contigs_i.append([start_i, i])
    contigs_t.append([comb.index.values[start_i], comb.index.values[i]])
    return contigs_i, contigs_t


def hours_from_timedelta(td):
    return td.total_seconds() / 3600.


def load_file(data_path, fname, PID_filter=[]):
    """Main file to load data and return a time series for a single patient."""
    d = read_file(data_path, fname)
    if d is None:
        return {'PID': fname[:-8]}, None, None
    elif (len(PID_filter)>0):
        if d['PID'].values[0] not in PID_filter:
            return {'PID': fname[:-8]}, None, None
    toco = d[d['Data Type'] == 'UA']
    fecg = d[d['Data Type'] == 'HR2']
    toco_ts = mkts(toco, 'toco')
    fecg_ts = mkts(fecg, 'fecg')
    errors = 0
    if (fecg_ts is None):
        comb = pd.DataFrame(toco_ts, columns=['toco']).sort_index()
        comb['fecg'] = np.nan
        errors += 1
    if (toco_ts is None):
        comb = pd.DataFrame(fecg_ts, columns=['fecg']).sort_index()
        comb['toco'] = np.nan
        errors += 1
    if errors == 0:
        comb = toco_ts.merge(fecg_ts, how='outer', on='wct').sort_index()
    if 'toco' in comb.columns and 'fecg' in comb.columns:
        clean_ts(comb)
    contigs, contigs_t = mkcontigs(comb)
    # capture total hours
    contig_hours = []
    sequences = []
    for contig in contigs:
        tmp = comb.iloc[contig[0]:contig[1]]
        start = tmp.index.min()
        contig_hours.append(hours_from_timedelta(tmp.index.max() - start))
        tmp['PID'] = d['PID'].values[0]
        tmp['Timestamp'] = tmp.index
#         tmp = tmp.set_index('PID')
        sequences.append(tmp)
    total_hours = np.sum(contig_hours)
    meta = {'contigs': contigs,
            'contigs_t': contigs_t,
            'PID': d['PID'].values[0],
            'start_date': comb.index.min(),
            'end_date': comb.index.max(),
            'contig_hours': contig_hours,
            'total_hours': total_hours,
            'toco_count': comb['toco'].notnull().sum(),
            'fecg_count': comb['fecg'].notnull().sum(),
           }
    return meta, comb, sequences


def plt_contigs(contigs, comb, fname):
    """Plot a single file in 1 hour spans per page for easy viewing."""
    with PdfPages(fname) as pdf:
        for contig in contigs:
            tmp = comb.iloc[contig[0]:contig[1]]
            start = tmp.index.min()
            hours = hours_from_timedelta(tmp.index.max() - start)
            for h in range(int(hours)):
                tmpp = tmp[(hours_from_timedelta(tmp.index - start) > h) & 
                           (hours_from_timedelta(tmp.index - start) <= (h+1))]
                fig, ax = plt.subplots(2,1,figsize=(15,4), sharex=True)
                ax[0].plot_date(tmpp.index, tmpp.toco, ',-',alpha=0.5,color='black')
                ax[0].set_ylim(0,100)
                ax[0].set_ylabel('toco')
                ax[0].grid(True)
                ax[0].set_title(f'{tmpp.index.min()}  -- hour {h} of {hours}')
                ax[1].plot_date(tmpp.index, tmpp.fecg,',-',alpha=0.5,color='black')
                ax[1].set_ylim(50,200)
                ax[1].set_ylabel('HR')
                ax[1].grid(True)
                pdf.savefig(fig)
                plt.close()

################################################################################
# load and filter samples
###############

def process_sample(
    f,
    data_path='.',
    horizon=20,
    window=60,
    sample_rate=4
    ):
    """Processess a single patient's time-series data."""
    # print(f)
    try:
        meta, s, seqs = load_file(data_path, f)
    except Exception as e:
        traceback.print_exc()
        return str("file load failure")

    if s.shape[0] < 1 or 'contigs' not in meta:
        return "Short / no sequence"
    if meta['toco_count'] != meta['fecg_count']:
        return "toco/fecg count mismatch"
    
    # clean duplicate time spots
    s = s.reset_index().drop_duplicates(subset='wct').set_index('wct')

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
    # if(s_trim.shape[0] < .7*raw_sample_window):
    #     return "Too much missingness"
    raw_sample_rate = 4
    # sample stride to hit sample rate
    sample_stride =  math.floor(raw_sample_rate/sample_rate) 

    if sample_rate != raw_sample_rate:
        #fill nans in missing samples 
        s_trim = s_trim.resample(pd.Timedelta(1/raw_sample_rate,'S')).asfreq()
        # smooth data using window corresponding to sample_rate Hz
        s_trim = s_trim.apply(lambda x: smooth(x, sample_stride+1, series=True))

        # downsample data to sample_rate Hz
        s_down = (
            s_trim.resample(pd.Timedelta(1/sample_rate,'S'), label='right')
            .apply(np.nanmean)
        )
        s_trim = s_down.copy()

        # Fill missing data with the previous value, then the next value if still missing
        for f in ['fecg','toco']:
            s_trim[f] = s_trim[f].fillna(method='ffill').fillna(method='bfill') 

    ts = dict(
        PID=meta['PID'],
        fecg = s_trim['fecg'].values,
        toco = s_trim['toco'].values,
        last_time = s_trim.index.max(),
        time = s_trim.index
    )
    # save
    return ts

def process_failures(seqs):
    ts_pass = [ts for ts in seqs if isinstance (ts, dict)]
    ts_fails = [ts for ts in seqs if isinstance (ts, str)]
    print(f'{len(ts_pass)} passes out of {len(seqs)}')
    print(f'{len(ts_fails)} failures out of {len(seqs)}')
    print(f'{len(seqs) - len(ts_pass) - len(ts_fails)} unexplained')
    cn = Counter(ts_fails)
    for k,v in cn.items():
        print(f"'{k}' removed {v} entries ({v/len(seqs)*100:.2f} %)")
    return ts_pass