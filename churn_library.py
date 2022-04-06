# library doc string
"""
Module to use churn functions

Author : Yoovraj
Date : March 2022
"""

# import libraries
import shap
import joblib
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # create new churn column and save histogram
    plt.clf()
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    df['Churn'].hist();
    plt.savefig('images/eda/churn_hist.png')
    
    # save histogram of customer age column
    plt.clf()
    df['Customer_Age'].hist();
    plt.savefig('images/eda/cust_age_hist.png')

    # save bar plot of marital status normalized values
    plt.clf()
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_bar.png')

    # save distribution plot of total trans ct values
    plt.clf()
    sns.displot(df['Total_Trans_Ct'])
    plt.savefig('images/eda/total_trans_ct.png')

    # save correlation heatmap of input features
    plt.clf()
    plt.figure(figsize=(40,20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('images/eda/correlation.png')
    

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # if the response string is empty, then use default "Churn" as the suffix
    if response:
        response = "Churn"
    
    # loop through each category for creating churn for that category
    for category in category_lst:
        category_churn_col_name = category + "_" + response
        category_lst = []
        category_groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            category_lst.append(category_groups.loc[val])
        df[category_churn_col_name] = category_lst
    
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    # if the response string is empty, then use default "Churn" as the suffix
    if response:
        response = "Churn"
    y = df[response]
    
    # train test split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig('images/results/random_forest_classification_report.png')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off');
    plt.savefig('images/results/logistic_regression_classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.clf()
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90);
    plt.savefig(output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    ## 1. Start training
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    ## 2. store model results: images + scores
    # store classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)
    
    # Store model roc curves
    plt.clf()
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp.plot(ax=ax)
    plt.savefig('images/results/model_scores.png')

    # Store shap plot
    plt.clf()
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig('images/results/model_shap_plot.png')

    ## 3. Store Model
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == "__main__":
    start_time = time.time()
    file_path = "./data/bank_data.csv"
    print("==========")
    print("Importing data ...")
    df = import_data(file_path)
    print(df.columns)
    end_time = time.time()
    print("Importing data [done] ", round(time.time()-start_time), " seconds")
    
    # perform eda 
    print("==========")
    start_time = time.time()
    print("Performing EDA ...")
    perform_eda(df)
    print("Performing EDA [done] ", round(time.time()-start_time), " seconds")

    # encoding and feature engineeing
    print("==========")
    start_time = time.time()
    print("Performing encoding and feature engineering ...")
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'                
    ]
    df = encoder_helper(df, category_lst=cat_columns, response="Churn")
    print(df.columns)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df, response="Churn")
    print("Performing encoding and feature engineering [done] ", round(time.time()-start_time), " seconds")

    print("==========")
    start_time = time.time()
    print("Training Logistic regression and random forest classifier models ...")
    train_models(X_train, X_test, y_train, y_test)
    print("Training Logistic regression and random forest classifier models [done] ", round(time.time()-start_time), " seconds")

    print("==========")
    start_time = time.time()
    print("Loading models ...")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')
    print("Loading models [done] ", round(time.time()-start_time), " seconds")

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
            'Income_Category_Churn', 'Card_Category_Churn']
    
    print("==========")
    start_time = time.time()
    print("Generating feature importance plots for both models ...")
    feature_importance_plot(rfc_model, X_data=df[keep_cols], output_pth='images/results/rfc_feature_importance.png')
    print("Generating feature importance plots for both models [done] ", round(time.time()-start_time), " seconds")