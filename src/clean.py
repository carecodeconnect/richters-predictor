import pandas as pd

def remove_age_na(df, na_value=995):
    df = df.copy()
    filter_age_995 = (df['age'] == na_value)
    df = df.loc[~filter_age_995]
    return df

def remove_age_old(df, age_limit=100):
    df = df.copy()
    filter_age_old = (df['age'] >= age_limit)
    df = df.loc[~filter_age_old]
    return df

def remove_high_floors(df, n_count_floor=3):
    df.copy()
    filter_count_floors = (df['count_floors_pre_eq'] > 3)
    df = df.loc[~filter_count_floors]
    return df
