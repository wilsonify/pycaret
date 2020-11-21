# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: hyraka_env
#     language: python
#     name: hyraka_env
# ---

# ### Predicting the best math grades of the ENEM 2016
#
# In this example, we will predict the math score in the ENEM 2016 Brazilian National Exam.
# The dataset was obtained from INEP, a department from the Brazilian Education Ministry. It contains data from the applicants for the 2016 National High School Exam and can be downloaded [here](https://www.kaggle.com/davispeixoto/codenation-enem2). Inside this dataset there are not only the exam results, but the social and economic context of the applicants. Check here data for description: [Enem 2016 Microdata](https://s3-us-west-1.amazonaws.com/acceleration-assets-highway/data-science/dicionario-de-dados.zip).
#
# You will have two datasets - train.csv e test.csv - for using to predict math scores (`NU_NOTA_MT`). But the train file is a good amount of data from ENEM 2016 and contains most of the columns for those who would like to do some EDA.
#
# This notebook was created using PyCaret 2.0 by [Simone Perazzoli](https://github.com/simoneperazzoli/). 
# Last updated : 04-08-2020

# ### Loading libraries

# +
# #!pip3 install pycaret==2.0

# +
import numpy as np
import pandas as pd
import pycaret
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.regression import *
from scipy.stats import kurtosis, skew

pd.set_option('display.max_columns',200)
# -

# checking pycaret version
from pycaret.utils import version
version()

# ### Loading datasets

# +
# Train dataset
df_train = pd.read_csv('train.csv')
# Test dataset
df_test = pd.read_csv('test.csv')

# Creating answer dataframe
answer = pd.DataFrame()
# -

# ### Data preprocessing

# Saving the registration number:
answer['NU_INSCRICAO'] = df_test['NU_INSCRICAO']

# Droping the registration number from train and test dataframes:
df_train.drop(['NU_INSCRICAO'], axis=1, inplace=True)
df_test.drop(['NU_INSCRICAO'], axis=1, inplace=True)

# Checking dataframe shape
df_train.shape, df_test.shape

# +
#By checking the shape of the datasets we can see that there are more columns in the training data than in the 
#test data, so we will use only the features that exist in the test dataframe to analyze and determine which 
#features we should use to make the prediction.

cols = list(df_test)
cols.append('NU_NOTA_MT')

train = df_train[cols]
test = df_test
# -

# Viewing training data:
train.head()

# Viewing test data:
test.head()

# Checking dataframe shape after transformation
train.shape, test.shape


# +
# Creating a funtion to summarize dataframe information

def data_summary(df):
    '''Summary dataframe information'''

    df = pd.DataFrame({'type': df.dtypes,
                       'amount': df.isna().sum(),
                       'null_values (%)': (df.isna().sum() / df.shape[0]) * 100,
                       'unique': df.nunique()})
    return df


# -

# Train summary:
data_summary(train)

# Test summary:
data_summary(test)

# #### Analysing target "NU_NOTA_MT"

# Checking the distribution of the variable:
plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,6))
sns.distplot(train.NU_NOTA_MT, bins=25)
plt.xlabel('Score')
plt.title('Distribuition of math scores');

# Descriptive statistics for target:
train['NU_NOTA_MT'].describe()

print(f'Kurtosis: {train.NU_NOTA_MT.kurt()}')
print(f'Asymmetry: {train.NU_NOTA_MT.skew()}')


# - **Kurtosis** is used to identify outliers in the distribution. Here, its value is > 3, which means that the distribution tails tend to be lighter than in normal distribution or, the lack of outliers.  
#
# - The positive **asymmetry** means that we have a slightly tail on the right side of the distribution. The data is moderately skewed as our asymmetry value is between 0.5 and 1.

# #### Data cleaning and transforming

# Creating a function to remove irrelevant features
def data_cleaning(df):
    '''Removing irrelevant features'''

    df.drop(['TP_DEPENDENCIA_ADM_ESC',
             'TP_ENSINO',
             'CO_PROVA_CN',
             'CO_PROVA_CH',
             'CO_PROVA_LC',
             'CO_PROVA_MT',
             'SG_UF_RESIDENCIA',
             'CO_UF_RESIDENCIA',
             'TP_NACIONALIDADE',
             'IN_BAIXA_VISAO',
             'IN_CEGUEIRA',
             'IN_SURDEZ',
             'IN_DISLEXIA',
             'IN_DISCALCULIA',
             'IN_SABATISTA',
             'IN_GESTANTE',
             'IN_IDOSO',
             'TP_ANO_CONCLUIU','TP_PRESENCA_CN',
             'TP_LINGUA','TP_PRESENCA_CH',
             'IN_TREINEIRO', 'TP_PRESENCA_LC',
             'TP_ST_CONCLUSAO',
             'TP_STATUS_REDACAO', 
             'NU_IDADE',
             'Q027'], axis=1, inplace=True)
    return df


# Cleaning data:
data_cleaning(train)
data_cleaning(test)
train.shape, test.shape


# - Here, the 3 first collumns were dropped due to present missing values >50%. The other ones were dropped after a manual analysis of the [data dictionary](https://s3-us-west-1.amazonaws.com/acceleration-assets-highway/data-science/dicionario-de-dados.zip).

# Creating a function to impute missing values:
def data_imputation(df):
    '''Imputing values to the missing data'''

    df.fillna(df.dtypes.replace({'float64': -100}), inplace=True)
    return df


data_imputation(train)
train.head();

data_imputation(test)
test.head()

# ### Data modeling with PyCaret

# ####  1- Seting up parameters:

# - `setup()` function initializes the environment in pycaret and creates the transformation pipeline to prepare the data for modeling and deployment. It must called before executing any other function and takes two mandatory parameters: dataframe {array-like, sparse matrix} and name of the target column. All other parameters are optional.
#
#
# - **data:** train data.
#
# - **target:** target feature.
#
# - **remove_multicollinearity:** When set to True, the variables with inter-correlations higher than the threshold defined under the multicollinearity_threshold are dropped. When two features are highly correlated with each other, the feature that is less correlated with the target variable is dropped.
#
# - **multicollinearity_threshold:** Threshold used for dropping the correlated features. 
#
# - **normalize:** When set to True, the feature space is transformed using the normalized_method param. Generally, linear algorithms perform better with normalized data however,  the results may vary and it is advised to run multiple experiments to evaluate the benefit of normalization. 
#
# - **normalize_method:** Defines the method to be used for normalization. 
#
# - **transform_target:** When set to True, target variable is transformed using the method defined in transform_target_method param. Target transformation is applied separately from feature transformations.
#
# - **session_id:** If None, a random seed is generated and returned in the Information grid. The unique number is then distributed as a seed in all functions used during the experiment. This can be used for later reproducibility of the entire experiment.

# Creating a pipeline to setup the model
pipeline = setup(data=train, target='NU_NOTA_MT', 
                 remove_multicollinearity=True,
                 normalize_method='robust',
                 multicollinearity_threshold=0.95, 
                 normalize=True, 
                 transform_target=True, 
                 session_id=1991)

# #### 2 - Comparing regression models

# - `compare_models()` function uses all models in the model library and scores them using K-fold Cross Validation. The output prints a score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold (default CV = 10 Folds) of all the available models in model library.

compare_models(fold=5)

# #### 3 - Creating a model

# - `create_model()` function creates a model and scores it using K-fold Cross Validation (default = 10 Fold). The output prints a score grid that shows MAE, MSE, RMSE, RMSLE, R2 and MAPE. This function returns a trained model object. 

model = create_model('xgboost', fold=5, round=2)

# #### 4 - Model Tunning

# - `tune_model()` function tunes the hyperparameters of a model and scores it using K-fold Cross Validation. The output prints the score grid that shows MAE, MSE, RMSE, R2, RMSLE and MAPE by fold (by default = 10 Folds). This function returns a trained model object.  

model = tune_model(model, fold=5)

# Checking score after cross-validation:
predict_model(model);

# Checking model parameters:
print(model)

# #### 5 - Analysing results

# Residuals Plot 
plot_model(model, plot='residuals')

# Prediction Error 
plot_model(model, plot='error')

# Cooks Distance Plot
plot_model(model, plot='cooks')

# Learning Curve
plot_model(model, plot='learning')

# Validation Curve
plot_model(model, plot='vc')

# Manifold Learning
plot_model(model, plot='manifold')

# Feature Importance
plot_model(model, plot='feature')


# Model Hyperparameter
plot_model(model, plot='parameter')

# #### 6 - Predicting math scores

# - `predict_model()` is used to predict new data using a trained estimator. 

predictions = predict_model(model, data=test, round=2)
predictions

# #### 7 - Finalize Model
#
# - `finalize_model()` function fits the estimator onto the complete dataset passed during the setup() stage. The purpose of this function is to prepare for final model deployment after experimentation.

final_model = finalize_model(model)

# #### 8 - Save Model
#
# - `save_model()` function saves the transformation pipeline and trained model object into the current active directory as a pickle file for later use.

save_model(model, 'xgboost_model_04082020')
