import pandas as pd 
import numpy as np
import torch 

def make_tensor(df, df_y=None, normal_type='mean', impute_type='zero', summary_path=None): 
    df_cont = df.select_dtypes(include='float64')
    df_disc = df.select_dtypes(exclude='float64')
    
    # Create Mask  
    df_mask = (~df.isnull()).astype(int)
    df_mask.columns = np.array(df_mask.columns) + ' mask'
    
    # Discretise 
    df_disc = fix_str(df_disc) # fixes inconsistent labeling
    df_disc = pd.get_dummies(df_disc)
    
    # Normalise  
    if normal_type == 'mean': 
        df_cont = (df_cont - df_cont.mean())/df_cont.std()
    elif normal_type == 'minmax': 
        df_cont = (df_cont - df_cont.min())/(df_cont.max() - df_cont.min())
    else:
        raise ValueError('normal_type not recognized')
        
    # Impute 
    if impute_type == 'zero': 
        df_cont = df_cont.replace(np.nan, 0)
    elif impute_type == 'forward': 
        df_cont = df_cont.groupby('stay').fillna(method='ffill')
        stay_mean = df_cont.groupby('stay').mean()
        df_cont = df_cont.fillna(stay_mean).fillna(0)
    else: 
        raise ValueError('impute_type not recognized')
        
    # To Tensor 
    df_combined = pd.concat([df_cont, df_disc, df_mask], axis=1)
    M = df_combined.values
    M = M.reshape(-1, len(df.index.levels[1].unique()), df_combined.shape[1])
    T = torch.Tensor(M)
    
    # Create Summary 
    if summary_path is not None: 
        df_summary = {} 
        df_summary['global_missingness'] = df.isnull().mean()
        df_summary['cont_statistics'] = df_cont.agg(['mean', 'std', 'min', 'max', 'median'])
        df_summary['column_names'] = np.array(df_combined.columns)
        df_summary['stay_id'] = np.array(df_combined.index.levels[0])
        import pickle 
        with open(summary_path, 'wb') as handle:
            pickle.dump(df_summary, handle)
    
    # Create tensor for y
    if df_y is not None: 
        df_y = df_y.set_index('stay') 
        assert set(df_y.index) == set(df_combined.index.levels[0]) 
        df_y = df_y.reindex(df_combined.index.levels[0])
        T_y =  torch.Tensor(df_y.values)
        return((T, T_y), df_combined.columns)
    else: 
        return(T, df_combined.columns)
    
def fix_str(df): 
    # Fixes inconsistent labeling
    for col in df.columns: 
        df.loc[:,col] = df.loc[:,col].str.replace('[1234567890.]', '') \
                                     .str.strip()\
                                     .str.lower()\
                                     .str.replace(' ', '_')\
                                     .replace('none', None)
                    
    df.loc[:,'Glascow coma scale motor response'] = \
            df.loc[:,'Glascow coma scale motor response']\
                .replace('abnorm_extensn', 'abnormal_extension')\
                .replace('abnorm_flexion', 'abnormal_flexion')

    df.loc[:,'Glascow coma scale verbal response'] = \
            df.loc[:,'Glascow coma scale verbal response']\
                .replace(['et/trach', 'no_response-ett'], 'no_response')\
                .replace('inapprop_words', 'inappropriate_words')\
                .replace('incomp_sounds', 'incomprehensible_sounds')
                    
    return df