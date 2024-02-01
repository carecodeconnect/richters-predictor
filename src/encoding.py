from scikit.preprocessing import OneHotEncoder
import pandas as pd 

def one_hot_encoder(df, columns=None, min_frequency=3000, max_categories=5, drop_original=True):
    """This function performs one-hot encoding of categorical features

    Arguments: 
    - df (pd.DataFrame): DataFrame with categorical features
    - columns (list of strings): List of features that should be one-hot encoded. If None, one-hot encode all columns with dtype=object
    - min_frequency (int): Minimum frequency of value. If below, put into "infrequent" class
    - max_categories (int): Maximum number of one-hot encoded features.
    - drop_original (boolean): if True, old columns will be dropped

    Returns:
    - df (pd.DataFrame): Dataframe with one-hot encoded features
    """
    if columns == None:
        df.select_dtypes(include="object").columns

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", 
                    min_frequency=min_frequency, max_categories=max_categories)

    df = df.copy()
    df.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(df[columns])
    if drop_original:
        df = df.drop(columns=columns)

    return df