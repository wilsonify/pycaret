# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # PIMA Model using PYCaret 1.0

# + active=""
# Context: - 
#
# This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.
#
#
# Business Problem: - 
# Build a machine learning model to accurately predict whether or not the patients in the dataset have diabetes or not?
#
# Acknowledgements
# Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.
#
# -

import pandas as pd

# +
#loading the data 
# -

dataset=pd.read_csv('datasets_228_482_diabetes.csv')

# + active=""
# Peeking into data we see our target is Outcome 1- Diabetic, 0-Not diabetic. All numeric data
# -

dataset.head()

dataset.shape # we have  768 rows and 9 data columns  


# # EDA 

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)
import squarify


# +
Diabetic = dataset[(dataset['Outcome'] != 0)]
Non_diabetic = dataset[(dataset['Outcome'] == 0)]


def target_count():
    trace = go.Bar( x = dataset['Outcome'].value_counts().values.tolist(), 
                    y = ['Non_diabetic','diabetic' ], 
                    orientation = 'h', 
                    text=dataset['Outcome'].value_counts().values.tolist(), 
                    textfont=dict(size=15),
                    textposition = 'auto',
                    opacity = 0.8,marker=dict(
                    color=['lightskyblue', 'gold'],
                    line=dict(color='#000000',width=1.5)))

    layout = dict(title =  'Outcome variable')

    fig = dict(data = [trace], layout=layout)
    py.iplot(fig)


# -

target_count()


# Dataset is clearly unbalanced we can use SMOTE sampling to balance the classes. But In this notebook we are not going to look at it. 

# +
# Visulazing the distibution of the data for every feature
plt.figure(figsize=(20, 20))

for i, column in enumerate(dataset.columns, 1):
    plt.subplot(3, 3, i)
    dataset[dataset["Outcome"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO', alpha=0.6)
    dataset[dataset["Outcome"] == 1][column].hist(bins=35, color='yellow', label='Have Diabetes = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)
# -

# Before doing anything the first thing we need to do is splitting data into training set and validation set. 
# 1)	Training set : - Data set on which we build the model  and fine tune the model. 
# 2)	Validation set: - Data set on which we test how well our finalized model is performing. It is important that during the modelling stage we don’t expose this data for our model. This should be unseen instance for our model.
#

# +
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index).reset_index(drop=True)
data.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions ' + str(data_unseen.shape))
# -

# I have 730 data points to model and 38 data points to test my model. 

from pycaret.classification import *

# Setup function in PyCaret is the most important function this is where we perform all our data preprocessing steps. 
# •	Data = Data for modelling 
# •	Target = Target column that we want to predict in this case it is diabetic or not 
# •	Session_id  = User defined session id 
# •	Normalization =  Machine learning models work well when the input features do not have huge variation such as BMI and  Glucose their values are on different scale. It is important to scale then hence we use normalize parameter 
# •	Transformation = While normalization reduces the variance transformation changes the data so that It could be represented in Gaussian distribution (normal curve).
# •	Multicollinearity: - When the data  is highly co-related our algorithms tend not to generalize very well so it is important to remove multi- collinearity by using the remove_multicollinearity  and multicollinearity_threshold parameters in setup
# •	Sometimes a dataset may have a categorical feature with multiple levels, where distribution of such levels are skewed and one level may dominate over other levels. This means there is not much variation in the information provided by such feature.  For a ML model, such feature may not add a lot of information and thus can be ignored for modeling. This can be achieved in PyCaret using ignore_low_variance parameter 
#

clf = setup(data = data, target = 'Outcome',session_id=1229,normalize=True,transformation=True,ignore_low_variance=True,
           remove_multicollinearity=True, multicollinearity_threshold=0.95)

# Before you proceed Make sure all your data types are inferred correctly If so press enter if not change the data types. You can find more info about data types that on this page https://pycaret.org/data-types/

compare_models(sort='AUC')

# + active=""
# In one line Pycaret fitted 14 models. I am using sort parameter and sorting by AUC because I am interested in optimizing AUC as my classifier metric. It is important to me that my classifier differentiates 1’s as 1’s and 0’s as 0’s. We can sort it by any metric choosing your metric depends on your business case. 
#
# Here I am picking CatBoost Classifer because it fared well with all metrics except AUC let's see if we can optimize it.  
# -

catboost = create_model('catboost',fold =10) #CatBoost Classifier

# + active=""
# catboost is giving us  82.27  AUC let’s see if we can improve on that by using tune_model 
# -

tuned_cat_boost= tune_model('catboost', optimize = 'AUC')

# Well we have improved from 82.27 tp 82.70 

# +
#Lets Create  more classifiers 
# -

lr = create_model('lr', fold =10)

tuned_lr= tune_model('lr',optimize='AUC') # tuned_logistic   81.6 AUC

gbc= create_model('gbc',fold =10) #  Gradient boosting  82.60

tuned_graident_bosting = tune_model('gbc',optimize='AUC') # tuned Gradient boosting 80.5 

xgb = create_model('xgboost',fold =10)

# +
# lets blend this above models and see if we can beat 82.7 cat boost 
# -

blend_specific_soft = blend_models(estimator_list = [lr,gbc,xgb], method = 'soft')

# +
# we did --- 83.52 
# -

plot_model(blend_specific_soft)

# alright lets make some predictions 
predict_model(blend_specific_soft)

# +
# Finalizing the model
# -

final_model = finalize_model(blend_specific_soft)

# +
# Moment of truth … let’s see how our classifier does if we can predict using the unseen data 
# -

unseen_predictions = predict_model(final_model, data=data_unseen)
unseen_predictions

# Label and score are added to the data frame. 
# •	Label is the predicted outcome 
# •	Score is the predicted probability 
#




