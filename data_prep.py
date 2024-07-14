import numpy as np
import pandas as pd

def get_lawschool_data(as_df=False):
    
    df = pd.read_csv('./data/lawschool.csv')
    df = df.dropna()
    y = df['ugpa'] #target: gpa in [0,4]
    df = df.drop('ugpa', axis=1)
    df['gender'] = df['gender'].map({'male': 1, 'female': 0})
    df_bar = df['bar1']
    df = df.drop('bar1', axis=1)
    df['bar1'] = [int(grade == 'P') for grade in df_bar]
    df['race'] = [int(race == 7.0) for race in df['race']] #setting S=1 for white, S=0 for non-white
    S = df['race'] #sensitive attribute
    df = df.drop('race', axis=1)
    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y

def get_communities_data(as_df=False):
    
    df = pd.read_csv('./data/communities.csv')
    df = df.fillna(0)

    sens_attrs = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    df['race'] = df[sens_attrs].idxmax(axis=1) #creating a new column based on ethnicity
    df = df.drop(columns=sens_attrs)

    df = df.drop(df[df['ViolentCrimesPerPop']==0].index)
    y = df['ViolentCrimesPerPop'] #target
    df = df.drop('ViolentCrimesPerPop', axis=1)

    mapping = {'racePctWhite':1, 'racepctblack':0, 'racePctAsian':0, 'racePctHisp':0} 

    S = df['race'].map(mapping) #sensitive attribute: S=1 for white, S=0 for non-white
    df = df.drop('race', axis=1)

    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y
    
def get_frequencies(S):
    p = []
    for p_s in sorted(S.value_counts(1)):
        p.append(p_s)
    return p 
    
    
#as in ADW

def get_adult_data(as_df=False):
    """
    Parse the entire dataset of adult
    """
    df = pd.read_csv("./data/adult.csv", )
    df = df.dropna()
    df = df.replace({'?':np.nan}).dropna()
    df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
    y = df["age"] #target
    df = df.drop("age", 1)
    # hot code categorical variables
    S = df['sex']
    df = df.drop('sex', axis=1)
    df = drop_str(df)
    log_numeric_features(df)
    X = df.to_numpy() #features
    
    if as_df: #for comparing with agarwal
        return df, S, y
    else:
        return X, S, y


def drop_str(df):
    cols = df.columns
    for c in cols:
        if isinstance(df[c][1], str):
            column = df[c]
            df = df.drop(c, 1)
    return df

def log_numeric_features(df):
    cols = df.columns
    for c in cols:
        column =df[c]
        unique_values = list(set(column))
        n = len(unique_values)
        if n > 2:
            df[c] = np.log(1 + df[c])


