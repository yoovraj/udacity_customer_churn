"""
Module to test functions

Author : Yoovraj
Date : April 2022
"""

# import libraries
import os
import random
import logging
import pandas as pd
import churn_library as cls

# define logging
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
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
        logging.error("Testing import_eda: The file was not found")
        raise err

    try:
        # check for non zero size of dataframe
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cls.import_data("./data/bank_data.csv")

    # call the function under test
    perform_eda(df)

    # set expectations for output files
    file_names = [
        'images/eda/churn_hist.png',
        'images/eda/cust_age_hist.png',
        'images/eda/marital_status_bar.png',
        'images/eda/total_trans_ct.png',
        'images/eda/correlation.png'
    ]
    try:
        # check if expected files exist
        assert all(os.path.isfile(file_name) for file_name in file_names)

        # check if expected files are not empty
        assert all(os.stat(file_name).st_size >
                   0 for file_name in file_names)

        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_eda: Some files were not found or empty %s",
                      list(filter(lambda x: not os.path.isfile(x), file_names)))
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # generate test dataset of shape (4,3)
    df = pd.DataFrame({
        "Col_1": [1, 2, 3, 4],
        "Col_2": [-1, -1, -1, 1],
        "Churn": [1, 1, 0, 0]
    })
    category_lst = [
        'Col_1'
    ]

    # call function under test
    df = encoder_helper(df, category_lst)

    try:
        # check for non zero dataframe
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert df.shape[0] == 4
        assert df.columns.size == 4
        assert 'Col_1_Churn' in df.columns.tolist()
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: returned dataframe is corrupted")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # generate test dataset of shape (10,5)
    df = pd.DataFrame({
        "Col_1": [random.randint(0, 10) for i in range(10)],
        "Col_2": [random.randint(0, 10) for i in range(10)],
        "Col_3": [random.randint(0, 10) for i in range(10)],
        "Col_4": [random.randint(0, 10) for i in range(10)],
        "Churn": [random.randint(0, 1) for i in range(10)],
    })
    keep_cols = [
        'Col_2',
        'Col_3'
    ]

    # call function under test
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, keep_cols)

    try:
        # check the correct split based on ration defined inside feature
        # engineering function
        assert len(X_train) == 7
        assert len(y_train) == 7
        assert len(X_test)  == 3
        assert len(y_test)  == 3
        # check if all the columns in keep_cols are present in returned data
        assert all(col in X_train.columns.tolist() for col in keep_cols)
        assert all(col in X_test.columns.tolist() for col in keep_cols)

    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: \
                    returned tuple of training and testing data is corrupted")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    # generate training data
    X_train = pd.DataFrame({
        'Col_1': [random.randint(0, 10) for i in range(70)],
        'Col_2': [random.randint(0, 10) for i in range(70)]
    })
    y_train = [random.randint(0, 1) for i in range(70)]

    # generate testing data
    X_test = pd.DataFrame({
        'Col_1': [random.randint(0, 10) for i in range(30)],
        'Col_2': [random.randint(0, 10) for i in range(30)]
    })
    y_test = [random.randint(0, 1) for i in range(30)]

    # call function under test
    train_models(X_train, X_test, y_train, y_test)

    # set expectations for output files
    file_names = [
        'images/results/random_forest_classification_report.png',
        'images/results/logistic_regression_classification_report.png',
        'images/results/model_scores.png',
        'images/results/model_shap_plot.png',
        './models/rfc_model.pkl',
        './models/logistic_model.pkl'
    ]
    try:

        # check if expected files exist
        assert all(os.path.isfile(file_name) for file_name in file_names)

        # check if expected files are not empty
        assert all(os.stat(file_name).st_size >
                   0 for file_name in file_names)

        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        logging.error("Testing train_models: Some files were not found or empty %s",
                      list(filter(lambda x: not os.path.isfile(x), file_names)))
        raise err


if __name__ == "__main__":
    # test all functions and report final result
    result = []
    try:
        test_import(cls.import_data)
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_eda(cls.perform_eda)
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_encoder_helper(cls.encoder_helper)
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_perform_feature_engineering(cls.perform_feature_engineering)
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    try:
        test_train_models(cls.train_models)
        result.append(True)
    except Exception as e:
        logging.error(e)
        result.append(False)

    passed_cases = len(list(filter(lambda x: x, result)))
    failed_cases = len(list(filter(lambda x: not x, result)))
    TOTAL_CASES = len(result)

    if all(result):
        # log success as final result
        logging.info("Final Test Result : Success %s/%s",
                     passed_cases, TOTAL_CASES
                     )
    else:
        # log failure as final result
        logging.error("Final Test Result : Failed %s/%s",
                      failed_cases, TOTAL_CASES
                      )
