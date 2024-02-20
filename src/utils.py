import glob
import os
from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import f1_score


DATA_DIR = Path('../data')
TRAIN_VALUES_FILE = 'train_values.csv'
TRAIN_LABELS_FILE = 'train_labels.csv'
TEST_VALUES_FILE = 'test_values.csv'

SUBMIT_DIR = Path('../submissions')
SUBMIT_TEMPLATE_FILE = 'submission_format.csv'

MODEL_DIR = Path('../models')


# Mehods for loading data
def load_train_data():
    """
    Load the training data from the local file or from the URL.

    Args:
        local: boolean, if True, load from local file, else from URL
    
    Returns:
        train_data: DataFrame with training data
    """
    # Define path to data files
    path_train_values = DATA_DIR / TRAIN_VALUES_FILE
    path_train_labels = DATA_DIR / TRAIN_LABELS_FILE
    
    # Read the training values and labels
    train_values = pd.read_csv(path_train_values, index_col='building_id')
    train_labels = pd.read_csv(path_train_labels, index_col='building_id')
    
    # Join training values and labels in one DataFrame
    train_data = train_values.join(train_labels)
    
    return train_data


def load_test_data():
    """
    Load the test data from the local file or from the URL.

    Args:
        local: boolean, if True, load from local file, else from URL

    Returns:
        test_data: DataFrame with test data
    """
    # Define path to test data file
    path_test_values = DATA_DIR / TEST_VALUES_FILE
    
    # Read the test set
    test_values = pd.read_csv(path_test_values, index_col='building_id')
    
    return test_values


# Methods for evaluating and saving models
def evaluate_model(pipe, X_train, X_valid, y_train, y_valid):
        """
        Evaluate the model using the F1 score (micro).

        Args:
            pipe: trained model
            X_train: training data
            X_valid: validation data
            y_train: training labels
            y_valid: validation labels

        Returns:
            score_valid: F1 score on validation data
            score_train: F1 score on training data
        """
        # Train model
        pipe.fit(X_train, y_train)
        # Make predictions on validation and training data
        preds_valid = pipe.predict(X_valid)
        preds_train = pipe.predict(X_train)
        # Calculate F1 score (micro) on validation and training data
        score_valid = f1_score(y_valid, preds_valid, average='micro')
        score_train = f1_score(y_train, preds_train, average='micro')

        return score_valid, score_train


# Methods for saving and loading models
def save_model(pipe, timestamp):
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
        print(f"WARNING: this file already exists!")
    else:
        pickle.dump(pipe, open(str(MODEL_DIR / filename), "wb"))

    return MODEL_DIR / filename


def load_model(file_path):
    """
#     Load a model from a pickle file.

#     Args:
#         file_path: path to the pickle file

#     Returns:
#         estimator: the trained model
#     """
    return pickle.load(open(file_path, "rb"))


# Methods for saving submissions
def _format_submission(pipe, test_data, label_encoder=None):
    """
    Format predictions into a DataFrame that matches the submission format.

    Args:
        estimator: trained model
        test_data: test data to make predictions on

    Returns:
        my_submission: DataFrame with predictions in the correct format
    """
    predictions = pipe.predict(test_data)
    
    if label_encoder:
        predictions = label_encoder.inverse_transform(predictions)
    
    submission_format = pd.read_csv(SUBMIT_DIR / SUBMIT_TEMPLATE_FILE, 
                                    index_col='building_id')
    
    my_submission = pd.DataFrame(data=predictions,
                                 columns=submission_format.columns,
                                 index=submission_format.index)
    
    return my_submission
    
    
def save_submission(pipe, test_data, timestamp, label_encoder=None):
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
    submission = _format_submission(pipe, test_data, label_encoder)
    
    # Create filename based on current timestamp
    filename = 'submission' + str(int(timestamp)) + '.csv'
    
    # Create submissions directory if it does not exist:
    if not os.path.exists(SUBMIT_DIR):
        os.mkdir(SUBMIT_DIR)

    # Check that filename does not already exist before saving
    filename_match = glob.glob(str(SUBMIT_DIR / filename))
    
    if filename_match:
        print(f"WARNING: this file already exists!")
    else:
        submission.to_csv(SUBMIT_DIR / filename)

    return SUBMIT_DIR / filename
    
    