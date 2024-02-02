from pathlib import Path
import glob
import os
from datetime import datetime
import pandas as pd
import pickle


SUBMIT_URL = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/submission_format.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240202%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240202T104321Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=58916a2bc5dda241e2f6897f39f882fd36e8cab642e0c1bf60ceaddf73de048f'
SUBMIT_DIR = Path('../submissions')
SUBMIT_FORMAT_TEMPLATE = 'submission_format.csv'

MODEL_DIR = Path('../models')


def make_predictions(estimator, test_data):
    predictions = estimator.predict(test_data)
    return predictions

def format_submission(estimator, test_data):
    predictions = predictions = make_predictions(estimator, test_data)
    submission_format = pd.read_csv(SUBMIT_DIR / SUBMIT_FORMAT_TEMPLATE, index_col='building_id')
    
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    
    return my_submission

def save_submission(estimator, test_data, timestamp):
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
    
def load_model(file_path):
    return pickle.load(open(file_path, "rb"))

# TODO: make sure buildin_id index match submission and preditions
# Refactor into class
# Increment submission filenames_i