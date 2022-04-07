# Predict Customer Churn

Project **Predict Customer Churn** of [ML DevOps Engineer Nanodegree Udacity](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821).  
This project identify credit card customers that are most likely to churn. The data used for this project is from Kaggle's [Credit Card customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code) project. Also this project implements best coding practices.


## Project Description
A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

I got this dataset from a website with the URL as https://leaps.analyttica.com/home. I have been using this for a while to get datasets and accordingly work on them to produce fruitful results. The site explains how to solve a particular business problem.

Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

We have only 16.07% of customers who have churned. Thus, it's a bit difficult to train our model to predict churning customers.

## Project Structure
    .
    ├── Guide.ipynb          # Given: Getting started and troubleshooting tips
    ├── churn_notebook.ipynb # Given: Contains the code to be refactored
    ├── churn_library.py     # Definition of the functions
    ├── churn_script_logging_and_tests.py # ToDo: Tests and logs
    ├── README.md            # Provides project overview, and instructions to use the code
    ├── data                 # Read this data
    │   └── bank_data.csv
    ├── images               # Store EDA results 
    │   ├── eda
    │   └── results
    ├── logs                 # Store logs
    └── models               # Store models

## Running Files
### Environment Setup
```bash
# create python3.6 conda environment with name py36
conda create -n py36 python=3.6

# activate the conda environment using the name py36
conda activate py36

# install all the requirements from project's requirements.txt
pip install -r requirements.txt

# necessary to enable py36 kernel in jupyter notebook
python -m ipykernel install --user --name py36 --display-name "py36"
```

### Running churn_library.py
```bash
> python churn_library.py
==========
Importing data ...
Index(['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
       'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'],
      dtype='object')
Importing data [done]  0  seconds
==========
Performing EDA ...
Performing EDA [done]  1  seconds
==========
Performing encoding and feature engineering ...
Index(['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
       'Dependent_count', 'Education_Level', 'Marital_Status',
       'Income_Category', 'Card_Category', 'Months_on_book',
       'Total_Relationship_Count', 'Months_Inactive_12_mon',
       'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
       'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
       'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
       'Churn', 'Gender_Churn', 'Education_Level_Churn',
       'Marital_Status_Churn', 'Income_Category_Churn', 'Card_Category_Churn'],
      dtype='object')
Performing encoding and feature engineering [done]  1  seconds
==========
Training Logistic regression and random forest classifier models ...
lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html.
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
Training Logistic regression and random forest classifier models [done]  277  seconds
==========
Loading models ...
Loading models [done]  0  seconds
==========
Generating feature importance plots for both models ...
Generating feature importance plots for both models [done]  0  seconds
```

### Running test files
```bash
> python churn_script_logging_and_tests.py
> cat logs/churn_library.log
root - INFO - Testing import_data: SUCCESS
root - INFO - Testing test_eda: SUCCESS
root - INFO - Testing encoder_helper: SUCCESS
root - INFO - Testing train_models: SUCCESS
root - INFO - Final Test Result : Success 5/5

## Failure case
root - ERROR - Testing import_eda: The file wasn not found
root - ERROR - [Errno 2] No such file or directory: './data/bank_data1.csv'
root - INFO - Testing test_eda: SUCCESS
root - INFO - Testing encoder_helper: SUCCESS
root - INFO - Testing train_models: SUCCESS
root - ERROR - Final Test Result : Failed 1/5
```
