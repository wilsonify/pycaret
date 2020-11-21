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

# + [markdown] colab_type="text" id="TwuMD7ffQ70c"
# # Utilizing GPU for training our model
# This notebook is an example of how to use GPU to train our model. The model implemented here is LightGBM (Light Gradient Boosting Machine). The features of LightGBM are:
#
# * Faster training speed and higher efficiency.
#
# * Lower memory usage.
#
# * Better accuracy.
#
# * Support of parallel and GPU learning.
#
# * Capable of handling large-scale data.
#
# LightGBM supports parallel and GPU learning, which is the reason why it's a great choice for Kagglers. The actual paper which introduced LightGBM used XGBoost as a baseline model and showed that LightGBM outperformed XGBoost on training time & the dataset size it can handle.
#
# This is the reason why this algorithm gain so much popularity in less time. 
#
# Here, we will use the follwoing dataset:
#
# Link to the competition: https://www.kaggle.com/mlg-ulb/creditcardfraud
# THe dataset has 284K rows and 31 features. Also, this is a highly imabalanced dataset. Let's check how much time does LightGBM actually takes to train itself on GPU.

# + colab={} colab_type="code" id="JsQcMBDJjmM-"
#Importing libraries

import pandas as pd
import numpy as np
import pycaret


# + [markdown] colab_type="text" id="zx-m_QyMk0X4"
# ## Loading the data

# + [markdown] colab_type="text" id="1a5iBP3Nk3DE"
# Dataset information: The datasets contains transactions made by credit cards in September 2013 by european cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
#
# It contains only numerical input variables which are the result of a PCA transformation.
#
# This dataset was a part of Kaggle Competition too, where the participants needed to predict wether the transaction was a fraud one or normal.
#
# Link to the competition: https://www.kaggle.com/mlg-ulb/creditcardfraud

# + colab={"base_uri": "https://localhost:8080/", "height": 216} colab_type="code" id="WnXMK-pOj7Yc" outputId="098882af-b7b8-4458-d5fb-6c350fc968a0"
df=pd.read_csv("/content/drive/My Drive/creditcard.csv")
df.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 33} colab_type="code" id="Fqex7yyBkU0k" outputId="23d47664-7951-430e-82c3-8a649a6eeb95"
df.shape

# + [markdown] colab_type="text" id="10VMelselhZp"
# Our dataset contains 24000 rows and 24 columns

# + colab={"base_uri": "https://localhost:8080/", "height": 552} colab_type="code" id="gtKOmecwkC1c" outputId="4bf99b69-a5f2-47de-83fd-40dd34169edf"
df.isnull().sum()

# + [markdown] colab_type="text" id="KIL4GyhQloaz"
# There are no null values as observed from above table

# + [markdown] colab_type="text" id="MLFyMW2Gppu7"
# Now, let's check for the count of positive and negative classes in our dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 278} colab_type="code" id="rCPuWVd8kKYc" outputId="3606d26b-bbdd-4805-f4f5-5edca872a14d"
df["Class"].value_counts().plot.bar(legend=None)

# + [markdown] colab_type="text" id="x7kKRuulE72M"
# This is a highly imablanced dataset.

# + [markdown] colab_type="text" id="hGF3OuZMq1ra"
# ## Problem with imbalanced dataset: 
# We need to deal with the dataset in a correct way. When we will train our model than our model will achieve high accuracy but our trained model will predict a negative class in maximum number of cases. So, we also need to keep in mind the precision and recall score in such scenarios. 
#
# This problem is predominant in scenarios where anomaly detection is crucial like electricity pilferage, fraudulent transactions in banks, identification of rare diseases, etc. In this situation, the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.
#
# This happens because Machine Learning Algorithms are usually designed to improve accuracy by reducing the error. Thus, they do not take into account the class distribution / proportion or balance of classes.
#
# This guide describes various approaches for solving such class imbalance problems using Pycaret. 

# + [markdown] colab_type="text" id="Uch03o6zsn-h"
# ## Prepairing the setup

# + colab={} colab_type="code" id="_gxUecGVsnAM"
from pycaret.classification import *


# + colab={"base_uri": "https://localhost:8080/", "height": 918, "referenced_widgets": ["8ee681b3cab9472288464fdf80693c6e", "5c1c4d93291641559f6d93b1892fea91", "5a6a4f0a323a4daa86102db105e3d74a", "ff10c25a56174172badfa0a4b2e3303e", "d9d4492982e34abb8bdddbe04215c1dd", "7ff6bda982ce4d8b8a8482062f4d90e7"]} colab_type="code" id="0MDZGQo6p9bC" outputId="83e78dd2-5f0e-41f5-c586-2820d8e2da5e"
clf=setup(data=df,target='Class',fix_imbalance=True) #fix_imbalance will automaticaaly fix the imbalanced dataset by oversampling using the SMOTE method.

# + [markdown] colab_type="text" id="nQyNyXctwAGx"
# ### SMOTE method: SMOTe is a technique based on nearest neighbors judged by Euclidean Distance between data points in feature space. There is a percentage of Over-Sampling which indicates the number of synthetic samples to be created and this percentage parameter of Over-sampling is always a multiple of 100.

# + colab={} colab_type="code" id="SU3zRpKcVgi0"
#Uncomment the following code to compare the performance of all the classification models


#compare_models() 

# + [markdown] colab_type="text" id="PBAqx-IuPGb5"
# ## We will choose the model with high precision because here we need to have a high precision than high accuracy or high recalls.
#
# This link will provide you some overview of precision and recall.
# Link: https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall

# + [markdown] colab_type="text" id="wNdaW3J5V3rk"
# We are creating the random forest classifier because it works really well with these types of dataset. You can have a quick view of the different models using 'compare_models()'

# + [markdown] colab_type="text" id="VyL_yFwjPxBl"
# # Using GPU for training our model
#
# Pycaret 2.0 now supports training XGBoost and LightGBM on the GPU. Here, we train a LightGBM(Light Gradient Boosting Machine) on the GPU and it only took around 50 seconds. Earlier training this model on CPU took around 50 minutes. 
# We just need to type the following code and execute it:
#
# >create_model('lightgbm', tree_method = 'gpu_hist', gpi_id = 0)
#
# For more information on LightGBM, the link is: https://lightgbm.readthedocs.io/en/latest/

# + colab={"base_uri": "https://localhost:8080/", "height": 390, "referenced_widgets": ["e42a59c320bb41a4909111c017f2a2c0", "8770095900034d968ea3bb7114f2aea1", "40a52b22f3c74f9cb92601a9335b11f6"]} colab_type="code" id="0RHd8qp1wL9Y" outputId="7577e9a5-b275-452e-99bd-f3c62f97e753"
import six
import sys
sys.modules['sklearn.externals.six'] = six

classifier=create_model('lightgbm', tree_method = 'gpu_hist', gpi_id = 0)
print(classifier)

# + [markdown] colab_type="text" id="aOH_EldwQ5Rt"
# ## Classification plots

# + colab={"base_uri": "https://localhost:8080/", "height": 401, "referenced_widgets": ["5ed480b9b8df4018a30e5be63fbbcf5f", "e99637a59a544b6badbf33007f55c873", "4679b95a7af6455ab2fcd4a73cdcf508"]} colab_type="code" id="kjbCMz7KQ43s" outputId="9167b52f-b68a-46e4-d2c1-e1d7eedace41"
# Plotting the classification report
plot_model(classifier,plot='class_report')


# + [markdown] colab_type="text" id="8ib3-Oc7X_6s"
# ### Here important point to notice is the precision, recall, and f1 score for the positive class that is '1'

# + colab={"base_uri": "https://localhost:8080/", "height": 374, "referenced_widgets": ["cc04386edc0e4715a360050759b97b4f", "737a822faa684390ac7aed76ce9ee3c5", "8f169763f51c48d789ffbb36203813d1"]} colab_type="code" id="ZdE1WS5cQ4q9" outputId="27d73fd2-68bb-4fdc-8224-5c880a85478a"
# Plotting the confusion matrix
plot_model(classifier,plot='confusion_matrix')


# + colab={} colab_type="code" id="vx-bh4yQXG7x"

