import pandas as pd 
import numpy as np

def generate_raw_timeseries(infile, outfile=None): 
    
    X = pd.read_csv(infile)
    
    # Simplify time axis to hourly intervals 
    max_hour = int(np.floor(max(X['Hours'])))
    X['Hours'] = np.floor(X['Hours']).astype(int)
    X.loc[X['Hours'] == max_hour, 'Hours'] = max_hour -1

    # Introduce paddings for hourly intervals when nothing happens
    padding_stay = np.repeat(X.stay.unique(), max_hour)
    padding_stay = np.expand_dims(padding_stay, axis=1)

    padding_hours = np.array(list(range(max_hour)) * len(X.stay.unique()))
    padding_hours = np.expand_dims(padding_hours, axis=1)

    padding_NAs = np.empty((len(X.stay.unique()) * max_hour, 17), dtype='object')

    padding_values = np.concatenate((padding_stay, padding_hours, padding_NAs), axis=1)

    padding_df = pd.DataFrame(padding_values)
    padding_df.columns = list(X.columns)

    # Aggregate data within the hour 
    X = pd.concat((X, padding_df), axis=0)

    X = X.set_index(['stay', 'Hours'])

    col_cont = X.columns[X.dtypes == 'float64']
    col_disc = X.columns[X.dtypes != 'float64']

    agg_dict = {key: 'mean' for key in col_cont}
    for key in col_disc: agg_dict[key] = 'first'

    X = X.groupby(['stay', 'Hours']).agg(agg_dict)
    assert X.shape[0]/max_hour == X.index.levels[0].shape[0]

    # Save hourly time-series
    if outfile is not None: 
        X.to_pickle(outfile)