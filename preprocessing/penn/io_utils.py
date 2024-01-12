import pandas as pd
import numpy as np

def read_file(data_path, fname):
    path = f'{data_path}{fname}'
    # print('loading',path)
    d = pd.read_csv(path, low_memory=True)
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
    if contigs:
        start_of_last_contig= comb.iloc[contigs[-1][0]:contigs[-1][1]].index.min()
    else:
        start_of_last_contig=0

    meta = {'contigs': contigs,
            'contigs_t': contigs_t,
            'PID': d['PID'].values[0],
            'start_date': comb.index.min(),
            'end_date': comb.index.max(),
            'start_of_last_contig':start_of_last_contig,
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