import glob
import os
from pathlib import Path
import pandas as pd
import pickle
from sklearn.metrics import f1_score


# Defining paths (file structure of repo)
DATA_DIR = Path('../data')
TRAIN_VALUES_FILE = 'train_values.csv'
TRAIN_LABELS_FILE = 'train_labels.csv'
TEST_VALUES_FILE = 'test_values.csv'

SUBMIT_DIR = Path('../submissions')
SUBMIT_TEMPLATE_FILE = 'submission_format.csv'

MODEL_DIR = Path('../models')


class Data:
    """Load implements methods for loading train and test data from files."""
    
    @staticmethod
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
    
    
    @staticmethod
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


class Model:
    """
    Model class used for training, evaluating and predicting
    """
    def __init__(self, estimator=None) -> None:
        self.estimator = estimator
        
        
    def train_model(self, X_train, y_train):
        """trains model
        Args:
            X_train: training data
            y_train. training labels
        """
        self.estimator = self.estimator.fit(X_train, y_train)
    
    
    def evaluate_model(self, X_train, X_valid, y_train, y_valid):
        """
        Evaluate the model using the F1 score.

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
        self.train_model(X_train, y_train)
        preds_valid = self.estimator.predict(X_valid)
        preds_train = self.estimator.predict(X_train)

        score_valid = f1_score(y_valid, preds_valid, average='micro')
        score_train = f1_score(y_train, preds_train, average='micro')

        return score_valid, score_train
    
        
    def make_predictions(self, data):
        """
        Make predictions using the estimator on the test_data.

        Args: 
            estimator: trained model
            test_data: test data to make predictions on

        Returns:
            predictions: array of predictions
        """
        #TODO: Warning if model is not yet trained
        
        predictions = self.estimator.predict(data)
        # XGB Classifiers assume that target is encoded starting from 0
        # if "XGB" in str(self.estimator["model"]):
        #     predictions += 1

        return predictions
    
    
    # @staticmethod
    def save_model(self, timestamp):
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
            pickle.dump(self.estimator, open(str(MODEL_DIR / filename), "wb"))

        return MODEL_DIR / filename
    
    
    def load_model(self, file_path):
        """
    #     Load a model from a pickle file.

    #     Args:
    #         file_path: path to the pickle file

    #     Returns:
    #         estimator: the trained model
    #     """
        self.estimator = pickle.load(open(file_path, "rb"))
        return self.estimator


class Submission:
    """
    Class for making and formatting submissions
    """
    def __init__(self, estimator, data):
        self.estimator = estimator
        self.data = data
    
    # @staticmethod
    def _format_submission(self):
        """
        Format predictions into a DataFrame that matches the submission format.

        Args:
            estimator: trained model
            test_data: test data to make predictions on

        Returns:
            my_submission: DataFrame with predictions in the correct format
        """
        predictions = Model.make_predictions(self.estimator, self.data)
        
        submission_format = pd.read_csv(SUBMIT_DIR / SUBMIT_TEMPLATE_FILE, 
                                        index_col='building_id')
        
        my_submission = pd.DataFrame(data=predictions,
                                    columns=submission_format.columns,
                                    index=submission_format.index)
        
        return my_submission
    
    
    # @staticmethod
    def save_submission(self, timestamp):
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
        submission = self._format_submission()
        
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
    
    