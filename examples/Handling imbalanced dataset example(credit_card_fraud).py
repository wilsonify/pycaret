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

# + colab={"base_uri": "https://localhost:8080/", "height": 216} colab_type="code" id="WnXMK-pOj7Yc" outputId="81fead06-4c27-4c93-d3b8-999caaf9ff14"
df=pd.read_csv("/content/drive/My Drive/creditcard.csv")
df.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 33} colab_type="code" id="Fqex7yyBkU0k" outputId="f5db4211-3266-4ac0-cbdd-f8553a55c1db"
df.shape

# + [markdown] colab_type="text" id="10VMelselhZp"
# Our dataset contains 24000 rows and 24 columns

# + colab={"base_uri": "https://localhost:8080/", "height": 552} colab_type="code" id="gtKOmecwkC1c" outputId="0ec8b121-5283-4fb0-e74b-8e110bfbdd51"
df.isnull().sum()

# + [markdown] colab_type="text" id="KIL4GyhQloaz"
# There are no null values as observed from above table

# + [markdown] colab_type="text" id="MLFyMW2Gppu7"
# Now, let's check for the count of positive and negative classes in our dataset

# + colab={"base_uri": "https://localhost:8080/", "height": 278} colab_type="code" id="rCPuWVd8kKYc" outputId="51bd3ee9-6c05-427a-bbcd-6c183a24cf82"
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


# + colab={"base_uri": "https://localhost:8080/", "height": 918, "referenced_widgets": ["0955e5b8be8042bbae4ee23c230a61eb", "7d901db567184b758cd7e56ab776f1d3", "9169f9e763314fddb72dfea8cabd09a4", "727e71d800a742bcb3c81ed2e8e964ee", "f9a6f7e7b553446392ca730a47f4bb3a", "a84f1abc70714e33a58147e1c6f995ef"]} colab_type="code" id="0MDZGQo6p9bC" outputId="32779414-a475-48a6-bba2-df9d97d9551f"
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

# + colab={"base_uri": "https://localhost:8080/", "height": 407, "referenced_widgets": ["ac8d355c4fd04396a4b3962e69b57e78", "e359da09f13c44939189b6dd7a8f7c7f", "bb8c34bfea704c3ca6f5733b4ed15f4d"]} colab_type="code" id="0RHd8qp1wL9Y" outputId="5571d088-6ef1-47fa-9377-a4f09258be91"
classifier=create_model('rf')
print(classifier)

# + [markdown] colab_type="text" id="aOH_EldwQ5Rt"
# ## Classification plots

# + colab={"base_uri": "https://localhost:8080/", "height": 401, "referenced_widgets": ["b974713bf81244b8ad8567b1b797b3a0", "1cc5d7149d994632a82330b094bd257f", "776d2b0cd8fc416790be305938018d61"]} colab_type="code" id="kjbCMz7KQ43s" outputId="ed8ab10d-6a8a-4e9e-8f2c-234b2d5c9548"
# Plotting the classification report
plot_model(classifier,plot='class_report')


# + [markdown] colab_type="text" id="8ib3-Oc7X_6s"
# ### Here important point to notice is the precision, recall, and f1 score for the positive class that is '1'

# + colab={"base_uri": "https://localhost:8080/", "height": 374, "referenced_widgets": ["33b14c76463243c89e55d95509cab12d"]} colab_type="code" id="ZdE1WS5cQ4q9" outputId="99306c54-e58b-40d3-c1b0-fe6a0956f314"
# Plotting the confusion matrix
plot_model(classifier,plot='confusion_matrix')


# + [markdown] colab_type="text" id="BPKk8BRnRWcO"
# ## Now, from the above plots we can easily conclude that we succesfully handled our highly imbalanced dataset as our precision score is high(90.1%). We would got a high accuracy score and recall score but a very low precision score, if we haven't succesfully handled our imbalanced dataset.

# + colab={} colab_type="code" id="vx-bh4yQXG7x"

