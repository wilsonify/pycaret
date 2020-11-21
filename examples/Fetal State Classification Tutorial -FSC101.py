# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# + id="eoSVEnEpORn6" colab_type="code" colab={}
## Importing necessary libraries 

import pandas as pd
import numpy as np
import warnings

# + [markdown] id="9ojSqYgIZf7s" colab_type="text"
# ## Dataset Introduction:
# Cardiotocography
# Since we all know that there are numerous techniques available to observe the fetus and ultrasound technique is one of the common ones but this ultrasound technique is not very helpful to record the heart-rate of the fetus and other details such as uterine contractions. This is where the cardiotocography comes into play. Cardiotocography is the technique that helps doctors to trace the heart rate of the fetus, which includes measuring accelerations, decelerations, and variability, with the help of uterine contractions. Further, this cardiotocography can be used to classify fetus into three states namely:
# * Normal trace
# * Suspicious trace
# * Pathological trace
#
# ## Problem Statement
# Fetal Pulse Rate and Uterine Contractions (UC) are among the basic and common diagnostic techniques to judge maternal and fetal well-being during pregnancy and before delivery. By observing the Cardiotocography data doctors can predict and observe the state of the fetus. Therefore we’ll use CTG data  to predict the state of the fetus. 
#
# Dataset link: https://www.kaggle.com/akshat0007/fetalhr

# + id="1pkwzPRqORn-" colab_type="code" colab={}
## Reading the dataset using pandas
df = pd.read_csv("CTG.csv")

# + id="nfMRHcmwORoC" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 216} outputId="5f96bdfc-07b6-40fc-df7c-3c38652ef15c"
## Having a look of our data
df.head()

# + [markdown] id="hc0WTN3jORoF" colab_type="text"
# # Feature Abbreviations used in the dataset :-
#
#
#
# ### FileName:	of CTG examination	
# ### Date:	of the examination	
# ### b:	start instant	
# ### e:	end instant	
# ### LBE:	baseline value (medical expert)	
# ### LB:	baseline value (SisPorto)	
# ### AC:	accelerations (SisPorto)	
# ### FM:	foetal movement (SisPorto)	
# ### UC:	uterine contractions (SisPorto)	
# ### ASTV:	percentage of time with abnormal short term variability  (SisPorto)	
# ### mSTV:	mean value of short term variability  (SisPorto)	
# ### ALTV:	percentage of time with abnormal long term variability  (SisPorto)	
# ### mLTV:	mean value of long term variability  (SisPorto)	
# ### DL:	light decelerations	
# ### DS:	severe decelerations	
# ### DP:	prolongued decelerations	
# ### DR:	repetitive decelerations	
# ### Width:	histogram width	
# ### Min:	low freq. of the histogram	
# ### Max:	high freq. of the histogram	
# ### Nmax:	number of histogram peaks	
# ### Nzeros:	number of histogram zeros	
# ### Mode:	histogram mode	
# ### Mean:	histogram mean	
# ### Median:	histogram median	
# ### Variance:	histogram variance	
# ### Tendency:	histogram tendency: -1=left assymetric; 0=symmetric; 1=right assymetric	
# ### A:	calm sleep	
# ### B:	REM sleep	
# ### C:	calm vigilance	
# ### D:	active vigilance	
# ### SH:	shift pattern (A or Susp with shifts)	
# ### AD:	accelerative/decelerative pattern (stress situation)	
# ### DE:	decelerative pattern (vagal stimulation)	
# ### LD:	largely decelerative pattern	
# ### FS:	flat-sinusoidal pattern (pathological state)	
# ### SUSP:	suspect pattern	
# ### CLASS:	Class code (1 to 10) for classes A to SUSP	
# ### NSP:	Normal=1; Suspect=2; Pathologic=3	
#

# + id="qS3F3sS9ORoG" colab_type="code" colab={}
## Dropping the columns which we don't need
df = df.drop(["FileName", "Date", "SegFile", "b", "e"], axis=1)

# + id="8_j3dXzCORoI" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 216} outputId="97ba0c46-aeb1-4170-be52-52bbfbc52df1"
df.head()

# + id="2lTTiQ0LORoL" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 100} outputId="cebda70b-1bcd-493d-d00a-d73857a37c5b"
df.columns

# + [markdown] id="gK6s6aXbRxlZ" colab_type="text"
# ## Performing some basic preprocessing techniques

# + id="W8hRcAhlORoQ" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 33} outputId="ab1046ff-8419-4aa2-a36c-2bcc92c759d3"
## This will print the number of columns and rows
print(df.shape)

# + id="QW-KthkFORoT" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 619} outputId="1ee8f939-b347-45c0-91cb-cb9d24422e3b"
## Checking for the null values
df.isnull().sum()

# + id="mJjIfadIORoW" colab_type="code" colab={}
## Dropping the the rows containing null values
df = df.dropna()

# + id="VojGGVKZORoY" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 619} outputId="528a202a-4a5a-43f2-c62c-604a5d7efb89"
df.isnull().sum()

# + id="6w2Qld4GORoa" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 619} outputId="317d255a-6a28-45b4-d6d4-9a7c4681ccd3"
## Checking the data type of the columns
df.dtypes

# + [markdown] id="vCVQdQ9ZR3Mv" colab_type="text"
# ## Importing the pycaret library

# + id="7oVTEnMxORod" colab_type="code" colab={}
# This command will basically import all the modules from pycaret that are necessary for classification tasks
from pycaret.classification import *

# + id="mmTt0qoaPb8B" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 918, "referenced_widgets": ["85013890037d4eae9e46bce44ef6fe31", "d1726335eb9847dcb8c408db31d29e39", "c069b042d8ee4808a04e5a02388dbc89", "8b11b69c85564a5fbd8a7656ff91e850", "2016bba888b7450f96b6018d8100e534", "d82f871fd9494cd29198d0758cc3ddde"]} outputId="9d1c6476-fffd-4d78-c678-b3a8c9ed1ace"
# Setting up the classifier
# Pass the complete dataset as data and the featured to be predicted as target
clf = setup(data=df, target='NSP')

# + id="5c5D_CcpP4pR" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 633, "referenced_widgets": ["04d55d6a1d1a482a9b7009586b58979b", "f09f1960e4254302a99b590ce733d608", "fff4b352b5594f77bf468695d468f288"]} outputId="a4a72143-5ccf-4429-f9cd-af1da1210ab0"
# This model will be used to compare all the model along with the cross validation
compare_models()

# + [markdown] id="jlbJqN8fS_fb" colab_type="text"
# ### The AUC score is 0.000 because it is not supported for the muli-classification tasks
#
# ### Also, from the above it is understood that Extreme Gradient Boosting(popularly known as XGBoost) model really performed well. So, we will proceed with Extreme Gradient Boosting model.

# + [markdown] id="fXcVr-yCTtBx" colab_type="text"
# ## Creating the Extreme Gradient Boosting(XGBoost) model

# + id="Rvy2ZevaRQB4" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["1deea41a50a34193a4a1d4a53ef32286", "aa38ef63b6d4402fabe4edb0d6419168", "ed7d18dacd964948995dc160f99736cf"]} outputId="05eac637-6b6a-45f4-cdfd-1d9e15008faf"
xgboost_classifier = create_model('xgboost')

# + id="ov9ni9O_TpsF" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 251} outputId="7e5e6e5b-664c-431f-bcc0-c1a60c517e75"
## Let's now check the model hyperparameters
print(xgboost_classifier)

# + [markdown] id="mdZAVDx2UPfN" colab_type="text"
# ## Tuning the hyperparametes for better performance

# + id="N2oq1DlCUKVU" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 273, "referenced_widgets": ["d73315df6042479586c9f7782ba8490b", "92b07214d46b4ae988829aeecd46b921", "7b59664e0dc74453889dbcde7ddd4978"]} outputId="05d577e0-1b52-467d-bc5a-f1cec63be28c"
# Whenenver we compare different models or build a model, the model uses deault
# hyperparameter values. Hence, we need to tune our model to get better performance

tuned_xgboost_classifier = tune_model(xgboost_classifier)

# + [markdown] id="nFWNkrBXU8Dr" colab_type="text"
# #### We can clearly conclude that our tuned model has performed better than our original model with default hyperparameters. The mean accuracy increased from 0.9899 to 0.9906
#
# #### pycaret library really makes the process of tuning hyperparameters easy
# #### We just need to pass the model in the following command
# #### tune_model(model_name)

# + [markdown] id="47ReN5fjWgyr" colab_type="text"
# ## Plotting classification plots

# + [markdown] id="ow6SMWB0Wy4s" colab_type="text"
# ## Classification Report

# + id="WqjMk3NAWfXf" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 401, "referenced_widgets": ["aec1a6ff8c24465d9a90d427594eb11f"]} outputId="e7bf4228-8280-4e92-e2ac-5a92138e7503"
plot_model(tuned_xgboost_classifier, plot='class_report')

# + [markdown] id="73pMQyL2W3kl" colab_type="text"
# ## Plotting the confusion matrix

# + id="iEuobdjkUplF" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 387, "referenced_widgets": ["6833484a36ba4c289e68a046d6c5f25e"]} outputId="09376b67-9df8-45ca-afc4-70099ffc7f50"
plot_model(tuned_xgboost_classifier, plot='confusion_matrix')

# + [markdown] id="qA09Ng-xYLum" colab_type="text"
# ## Saving the model for future predictions

# + id="DFCJOP3tXDzd" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 33} outputId="f9102521-8e3d-49a3-ef63-c3ef325aa917"
## This can be used to save our trained model for future use.
save_model(tuned_xgboost_classifier, "XGBOOST CLASSIFIER")

# + [markdown] id="lJTv2sxhYXbr" colab_type="text"
# ## Loading the saved model

# + id="F9JCVRfVYWDK" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 33} outputId="e3897b2d-09da-4339-fc2c-0384f6eb4cfa"
## This can be used to load our model. We don't need to train our model again and again.
saved_model = load_model('XGBOOST CLASSIFIER')

# + id="BrS9puqeYe-r" colab_type="code" colab={}

