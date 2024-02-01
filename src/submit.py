from pathlib import Path
import pandas as pd

SUBMIT_URL = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/submission_format.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2546ac2508675ca4e0161409520fab1e3552e9342e03468074572960192fc12c'
SUBMIT_DIR = Path('../submissions')


def make_predictions(estimator, test_data):
    predictions = estimator.predict(test_data)
    return predictions

def format_submission(SUBMIT_URL, predictions):
    submission_format = pd.read_csv(SUBMIT_URL, index_col='building_id')
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    return my_submission

def save_submission(estimator, test_data, filename):
    predictions = make_predictions(estimator, test_data)
    submission = format_submission(SUBMIT_URL, predictions)
    submission.to_csv(SUBMIT_DIR / filename)
    
# TODO: make sure buildin_id index match submission and preditions
# ADD function to create submission folder if not exist
# Refactor into class
# Increment submission filenames_i