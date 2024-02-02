from sklearn.preprocessing import OneHotEncoder, TargetEncoder
import pandas as pd 

def one_hot_encoder(df_train, df_test, 
                    columns=None, 
                    min_frequency=3000, max_categories=5, 
                    drop_original=True):
    
    """This function performs one-hot encoding of categorical features

    Arguments: 
    - df_train (pd.DataFrame): Train DataFrame with categorical features
    - df_test (pd.DataFrame): Test DataFrame with categorical features
    - columns (list of strings): List of features that should be one-hot encoded. If None, one-hot encode all columns with dtype=object
    - min_frequency (int): Minimum frequency of value. If below, put into "infrequent" class
    - max_categories (int): Maximum number of one-hot encoded features.
    - drop_original (boolean): if True, old columns will be dropped

    Returns:
    - df_train (pd.DataFrame): Train Dataframe with one-hot encoded features
    - df_test (pd.DataFrame): Train Dataframe with one-hot encoded features
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    if columns == None:
        columns = df_train.select_dtypes(include="object").columns
    
    # Create encoder instance
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", 
                    min_frequency=min_frequency, max_categories=max_categories)

    # Do encoding
    df_train.loc[:, ohe.get_feature_names_out()] = ohe.fit_transform(df_train[columns])
    df_test.loc[:, ohe.get_feature_names_out()] = ohe.transform(df_test[columns])


    if drop_original:
        df_train = df_train.drop(columns=columns)
        df_test = df_test.drop(columns=columns)

    return df_train, df_test



def target_encoder(df_train, df_test,
                    columns=None, column_target="damage_grade"):
    
    """This function performs target encoding of features

        Arguments: 
        - df_train (pd.DataFrame): Train DataFrame 
        - df_test (pd.DataFrame): Test DataFrame 
        - columns (list of strings): List of features that should be target encoded
        - column_target (string): target column

        Returns:
        - df_train (pd.DataFrame): Train Dataframe with target encoded features
        - df_test (pd.DataFrame): Test Dataframe with target encoded features
        """
    df_train = df_train.copy()
    df_test = df_train.copy()

    if columns==None:
        return df_train, df_test
    
    te = TargetEncoder(target_type="continuous")
    df_train.loc[:, columns] = te.fit_transform(df_train[columns], df_train[column_target])
    df_test.loc[:, columns] = te.transform(df_test[columns])

    return df_train, df_test
