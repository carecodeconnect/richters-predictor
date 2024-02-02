from pathlib import Path
import glob
import os
from datetime import datetime
import pandas as pd
import pickle

DATA_DIR = Path('../data')

URL_TRAIN_VALUES = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/train_values.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=be3146653c1a73b442dce121ce46ccef21a67a28347eefc2c62c473eff0a00e8'
URL_TRAIN_LABELS = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/train_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=36d24a38b88f5c6a4acddcbbc1db9b6518b71e50ec03a74fbe84fc30a62a9acc'
URL_TEST_VALUES = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/test_values.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5782f88e43df91a5b05951c6c868b1b9699b281042a38066e35aacba349e329c'

TRAIN_VALUES_FILE = 'train_values.csv'
TRAIN_LABELS_FILE = 'train_labels.csv'
TEST_VALUES_FILE = 'test_values.csv'

SUBMIT_URL = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/submission_format.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240202%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240202T104321Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=58916a2bc5dda241e2f6897f39f882fd36e8cab642e0c1bf60ceaddf73de048f'
SUBMIT_DIR = Path('../submissions')
SUBMIT_FORMAT_TEMPLATE = 'submission_format.csv'

MODEL_DIR = Path('../models')

def load_train_data(local: bool = False):
    """
    Load the training data from the local file or from the URL.

    Args:
        local: boolean, if True, load from local file, else from URL
    
    Returns:
        train_data: DataFrame with training data
    """
    if local:
        path_train_values = DATA_DIR / TRAIN_VALUES_FILE
        path_train_labels = DATA_DIR / TRAIN_LABELS_FILE
    else:
        path_train_values = URL_TRAIN_VALUES
        path_train_labels = URL_TRAIN_LABELS
    
    # Read the training values and labels
    train_values = pd.read_csv(path_train_values, index_col='building_id')
    train_labels = pd.read_csv(path_train_labels, index_col='building_id')
    
    # Join training values and labels in one DataFrame
    train_data = train_values.join(train_labels)
    
    return train_data

# Define function to load training data
def load_test_data(local: bool = False):
    """
    Load the test data from the local file or from the URL.

    Args:
        local: boolean, if True, load from local file, else from URL

    Returns:
        test_data: DataFrame with test data
    """
    if local:
        path_test_values = DATA_DIR / TEST_VALUES_FILE
    else:
        path_test_values = URL_TEST_VALUES    
    
    # Read the test values
    test_values = pd.read_csv(path_test_values, index_col='building_id')
    
    return test_values

def load_model(file_path):
    """
    Load a model from a pickle file.

    Args:
        file_path: path to the pickle file

    Returns:
        estimator: the trained model
    """
    return pickle.load(open(file_path, "rb"))

def make_predictions(estimator, test_data):
    """
    Make predictions using the estimator on the test_data.

    Args: 
        estimator: trained model
        test_data: test data to make predictions on

    Returns:
        predictions: array of predictions
    """
    predictions = estimator.predict(test_data)
    return predictions

def format_submission(estimator, test_data):
    """
    Format predictions into a DataFrame that matches the submission format.

    Args:
        estimator: trained model
        test_data: test data to make predictions on

    Returns:
        my_submission: DataFrame with predictions in the correct format
    """
    predictions = predictions = make_predictions(estimator, test_data)
    submission_format = pd.read_csv(SUBMIT_DIR / SUBMIT_FORMAT_TEMPLATE, index_col='building_id')
    
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    
    return my_submission

def save_submission(estimator, test_data, timestamp):
    """
    Save the submission to a csv file.

    Args:
        estimator: trained model
        test_data: test data to make predictions on
        timestamp: current timestamp
    
    Returns:
        None
    """
    # predictions = make_predictions(estimator, test_data)
    submission = format_submission(estimator, test_data)
    
    # Create filename based on current timestamp
    filename = 'submission' + str(int(timestamp)) + '.csv'
    
    # Create submissions directory if it does not exist:
    if not os.path.exists(SUBMIT_DIR):
        os.mkdir(SUBMIT_DIR)

    # Check that filename does not already exist before saving
    filename_match = glob.glob(str(SUBMIT_DIR / filename))
    
    if filename_match:
        print(f"WARNING: this file already exists! Try again in a few seconds")
    else:
        submission.to_csv(SUBMIT_DIR / filename)

def save_model(estimator, timestamp):
    """
    Save the model to a pickle file.

    Args:
        estimator: trained model
        timestamp: current timestamp
    
    Returns:
        None
    """
    # Create filename based on current timestamp
    filename = 'model_' + str(int(timestamp)) + '.pickle'

    # Create model directory if it does not exist:
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Check that filename does not already exist before saving
    filename_match = glob.glob(str(MODEL_DIR / filename))

    if filename_match:
        print(f"WARNING: this file already exists! Try again in a few seconds")
    else:
        pickle.dump(estimator, open(str(MODEL_DIR / filename), "wb"))

    return MODEL_DIR / filename
    


# TODO: make sure buildin_id index match submission and preditions
# Refactor into class
# Increment submission filenames_i