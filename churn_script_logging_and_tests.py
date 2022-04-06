import os
import logging
import churn_library as cls
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")
    perform_eda(df)
    file_names = [
        'images/eda/churn_hist.png',
        'images/eda/cust_age_hist.png',
        'images/eda/marital_status_bar.png',
        'images/eda/total_trans_ct.png',
        'images/eda/correlation.png'
    ]
    try:
        assert all([os.path.isfile(file_name) for file_name in file_names])
        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_eda: Some files were not found {}"
                    .format(list(filter(lambda x : not os.path.isfile(x), file_names))))
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    try:
        test_import(cls.import_data)
        test_eda(cls.perform_eda)
        test_encoder_helper(cls.encoder_helper)
    except Exception:
        logging.error("Final Test Result : Failed")








