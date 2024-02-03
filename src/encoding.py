import pandas as pd

cols_frequency_encode = ['geo_level_2_id', 'geo_level_3_id']

def freq_encode(df):
    """
    Frequency encode the columns in the list cols_frequency_encode.

    Args:
        df: DataFrame to encode

    Returns:
        DataFrame with frequency encoded columns
    """
    return pd.concat([df[col].map(df[col].value_counts(normalize=True)) for col in cols_frequency_encode], axis=1)

def get_house_volume(df):
    """
    Get the volume of the house.

    Args:
        df: DataFrame with columns area_percentage and height_percentage

    Returns:
        DataFrame with a new column house_volume
    """
    df.loc[:, "house_volume"] = df["area_percentage"] * df["height_percentage"]
    return df