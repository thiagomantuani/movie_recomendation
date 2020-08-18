
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import unidecode

def preprocessing():
    
    df = pd.read_csv('new_movies.csv')    
    df.drop_duplicates(inplace=True)
    
    df.drop(columns=['actors','certificate'],inplace=True)
        
    df = df[~df.year.isnull()]
    df = df.dropna(subset=['rating'])

    df['runtime'].fillna(0,inplace=True)
    df['gen_pred'] = df['genre'].str.split(',',n=1,expand=True)[0]
    df['year'] = df['year'].str.extract(r'(\d{4})',expand=True)
    df['runtime'] = df['runtime'].str.extract(r'(^([0-9][0-9]{0,2}|999))')
    df['votes'] = df['votes'].str.replace(',','',regex=True)
    df.drop(columns=['genre','sinopse_br','gen_pred'],inplace=True)
      
    df['runtime'].fillna(0,inplace=True)
        
    cols = ['rating','runtime','votes','year']
    for c in cols:
        df[c] = StandardScaler().fit_transform(df[c].values.reshape(-1,1))
    
    df['title'] = df.apply(lambda row: unidecode.unidecode(row['title']),axis=1)
    df['sinopse'] = df.apply(lambda row: unidecode.unidecode(row['sinopse']),axis=1)    
    
    df = df.sample(frac=1)
    
    return df

