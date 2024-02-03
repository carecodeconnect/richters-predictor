import pandas as pd


def freq_encode(df, cols_to_encode):
    return pd.concat([df[col].map(df[col].value_counts(normalize=True)) for col in cols_to_encode], axis=1)