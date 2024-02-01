import pandas as pd
from pathlib import Path

URL_TRAIN_VALUES = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/train_values.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=be3146653c1a73b442dce121ce46ccef21a67a28347eefc2c62c473eff0a00e8'
URL_TRAIN_LABELS = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/train_labels.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=36d24a38b88f5c6a4acddcbbc1db9b6518b71e50ec03a74fbe84fc30a62a9acc'
URL_TEST_VALUES = 'https://drivendata-prod.s3.amazonaws.com/data/57/public/test_values.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARVBOBDCYQTZTLQOS%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T103937Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5782f88e43df91a5b05951c6c868b1b9699b281042a38066e35aacba349e329c'

DATA_DIR = Path('../data')
TRAIN_VALUES_FILE = 'train_values.csv'
TRAIN_LABELS_FILE = 'train_labels.csv'
TEST_VALUES_FILE = 'test_values.csv'

# Define function to load training data from local directory or from URL
def load_train_data(local: bool = False):
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
    if local:
        path_test_values = DATA_DIR / TEST_VALUES_FILE
    else:
        path_test_values = URL_TEST_VALUES    
    
    # Read the test values
    test_values = pd.read_csv(path_test_values, index_col='building_id')
    
    return test_values