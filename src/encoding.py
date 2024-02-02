import pandas as pd

cols_frequency_encode = ['geo_level_2_id', 'geo_level_3_id']

def freq_encode(df):
    return pd.concat([df[col].map(df[col].value_counts(normalize=True)) for col in cols_frequency_encode], axis=1)

def get_house_volume(df):
    df.loc[:, "house_volume"] = df["area_percentage"] * df["height_percentage"]
    return df