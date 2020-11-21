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

# + [markdown] colab_type="text" id="KKRvdotGoCVs"
# #Data Set Overivew
#
# The tutorial uses the [Iowa Liquor Retails Sales](https://console.cloud.google.com/marketplace/details/iowa-department-of-commerce/iowa-liquor-sales). We will be using the dataset to predict future sales for one of the stores
#
# This dataset contains every wholesale purchase of liquor in the State of Iowa by retailers for sale to individuals since January 1, 2012. 
#
# The State of Iowa controls the wholesale distribution of liquor intended for retail sale, which means this dataset offers a complete view of retail liquor sales in the entire state. The dataset contains every wholesale order of liquor by all grocery stores, liquor stores, convenience stores, etc., with details about the store and location, the exact liquor brand and size, and the number of bottles ordered.

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="oxappFRWR6qm" outputId="014a4b8e-162f-4ef2-8ea2-adf75fafc0a1"
#importing necessary packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import numpy as np 
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import imageio
import os
from statsmodels.graphics.tsaplots import plot_acf



# + [markdown] colab_type="text" id="MI44fdd8Jtt5"
# # Reading Data
#

# + [markdown] colab_type="text" id="zp4xNDv3o5wy"
# ## Using GCP and Biq Query
# To setup a project on Google Cloud Platform and use Big Query go to: http://console.cloud.google.com
#
# You can also watch my tutorial here: https://www.youtube.com/watch?v=m5qQ5GLmcZs

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="UnJynIZ72ypg" outputId="054666ca-31f3-4591-c022-dc82b3d68663"
from google.colab import auth
auth.authenticate_user()
print('Authenticated')



# + [markdown] colab_type="text" id="cMa6SVLgp9wD"
# ## Pulling data for one store. 
# This dataset has quite a few dimensions, however, we will focus on just part for now which is just one store and for sales that have occured after 1st Jan 2018

# + colab={} colab_type="code" id="qeVGiMch8Wnb"
# %%bigquery --project bold-sorter-281506 df2  # bold-sorter-281506 df2 is the project id ; df2 is the dataframe name
SELECT *
FROM `bigquery-public-data.iowa_liquor_sales.sales`
where store_number  = '2633'
and date > '2018-01-01'

# + [markdown] colab_type="text" id="W3hSsmL0qOsj"
# ## Using a direct link
# A version of this dataset is also saved on my google drive. We can use it to pull the dataset
#

# + colab={} colab_type="code" id="jCqoAXcjISWe"
import pandas as pd

url = 'https://drive.google.com/file/d/1g3UG_SWLEqn4rMuYCpTHqPlF0vnIDRDB/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df2 = pd.read_csv(path)

# + colab={"base_uri": "https://localhost:8080/", "height": 37} colab_type="code" id="dMhpB4RxIp7i" outputId="212b9ba2-d2aa-4518-b853-b9dc18ed74bd"
path #save this path, just in case

# + [markdown] colab_type="text" id="9js2qCkSq5Dd"
# # Data Overview

# + colab={"base_uri": "https://localhost:8080/", "height": 394} colab_type="code" id="sUNtTaXH_zod" outputId="53af867d-3d65-4bdf-909f-7ef50ff396ee"
df2.head(5)

# + colab={} colab_type="code" id="mpDyU--L3Wxb"
df2_ds = df2[['date','sale_dollars']] # selecting the needed columns


# + colab={} colab_type="code" id="H0pw8xouAEPo"
df2_ds=df2_ds.sort_index(axis=0)

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="YbKw9JvyCjJn" outputId="3a7ac0f0-b7b0-4de5-fd2f-7fe21145526b"
df2_ds.tail(5)

# + colab={} colab_type="code" id="FlVg7MRD6zm4"
aggregated=df2_ds.groupby('date',as_index=True).sum()

# + colab={"base_uri": "https://localhost:8080/", "height": 51} colab_type="code" id="1Pph6mGw7K4A" outputId="9e3d206a-851c-410f-c0a8-f05c5750e067"
print(min(aggregated.index))
print(max(aggregated.index))

# + colab={} colab_type="code" id="bez5t6XJ0Y59"
aggregated.index=pd.to_datetime(aggregated.index)


# + [markdown] colab_type="text" id="2YYxAbSAAMc5"
# #Create Fetaures
# There are multiple ways of creating features, however, we will explore simpler ones - There are a few others, which I have commented for now

# + colab={} colab_type="code" id="ANeLnv9IAvgz"
def create_features(df):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['flag'] = pd.Series(np.where(df['date'] >= np.datetime64('2020-03-03'), 1, 0), index=df.index) #flag for COVID-19
    #df['rolling_mean_7'] = df['sale_dollars'].shift(7).rolling(window=7).mean()
    #df['lag_7'] = df['sale_dollars'].shift(7)
    #df['lag_15']=df['sale_dollars'].shift(15)
    #df['lag_last_year']=df['sale_dollars'].shift(52).rolling(window=15).mean()
  
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','flag','sale_dollars']]
    X.index=df.index
    return X


# + colab={} colab_type="code" id="Pq0YTx94Axo8"
def split_data(data, split_date):
    return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()


# + colab={"base_uri": "https://localhost:8080/", "height": 606} colab_type="code" id="Yd5OdAW5BDWv" outputId="2e4666f8-b896-488f-f5dc-3853ea4c8014"
aggregated=create_features(aggregated)
train, test = split_data(aggregated, '2020-06-15') # splitting the data for training before 15th June

plt.figure(figsize=(20,10))
plt.xlabel('date')
plt.ylabel('sales')
plt.plot(train.index,train['sale_dollars'],label='train')
plt.plot(test.index,test['sale_dollars'],label='test')
plt.legend()
plt.show()


# + [markdown] colab_type="text" id="_r8WQT9H-nL9"
# There is a lot of variation within the date, also, the dates are not continous, that is, there are gaps - we can do two things here, impute missing date or let it be. A major reason we will not create missing dates is because we are considering this data for predictive modeling rather than time series forecasting - hence the data is not depenent on the immediate past but the relationship of the features with sales over time

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="59jHH0I_CBHh" outputId="3e760938-bd67-411a-f99a-d0d664ad3021"
train.tail(4)

# + [markdown] colab_type="text" id="sNwbBhkkipSL"
# # Run PyCaret

# + colab={} colab_type="code" id="8hz1b-ViELta"
# #!pip install pycaret 

# + colab={} colab_type="code" id="0G24V-M4EWzs"
from pycaret.regression import *

# + [markdown] colab_type="text" id="lA_XyhCt_7K9"
# Setting up the model is extremely easy

# + colab={"base_uri": "https://localhost:8080/", "height": 956, "referenced_widgets": ["fe4ff1a8494c4db18b6f62cc48ebdf33", "cc1f9d5ed0fc4df8b06cc33fcfb12433", "a6657fb7e6c74968946ddf3c13aa852b"]} colab_type="code" id="Pb1KuIswE4Q-" outputId="56e01544-b229-47b7-9ac6-ed33061293f5"
reg = setup(data = train, 
             target = 'sale_dollars',
             numeric_imputation = 'mean',
             categorical_features = ['dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear',
                                     'flag']  , 
            transformation = True, transform_target = True, 
                  combine_rare_levels = True, rare_level_threshold = 0.1,
                  remove_multicollinearity = True, multicollinearity_threshold = 0.95, 
             silent = True)

# + [markdown] colab_type="text" id="9aqlIk56AauE"
# As a data scientist, I can't emphasize more on the usefulness of the function below - instead of pulling every single model, we just need one line to compare 20 different models! **This is insane!**

# + colab={} colab_type="code" id="2i0TiH95F6E-"
# returns best models - takes a little time to run
top3 = compare_models(n_select = 3)

# + [markdown] colab_type="text" id="qAa4eHGtAPF2"
# ## Creating baseline model

# + colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["ef1b8c8523d14bae8063cc50962be563", "32735450faed4860bba41860067140e7", "d0ddb74c942f4fbba405cc4d74ed9bcf"]} colab_type="code" id="BHH6Yiq1GaQo" outputId="58e43d60-f1ef-4745-e525-a197eca55bc3"
#we create a model using light gbm
lightgbm = create_model('lightgbm')

# + [markdown] colab_type="text" id="IB0PkaSHBotE"
# Being able to tune seamlessly and hardly writing a line is extremely useful

# + colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["a88069478f044e8b86163623106c6e16", "a6aaf5a7b8e94654bd9b734bfa90ae06", "33db443f1200442281a5064b139386cc"]} colab_type="code" id="WYO3g_5tJ0ze" outputId="30bedc3f-475a-47b6-f599-4a85952c84c5"
tuned_lightgbm = tune_model(lightgbm) 

# + colab={"base_uri": "https://localhost:8080/", "height": 376, "referenced_widgets": ["587e993da97d40e3929f3115a581afb0", "b05841eeccb641e4bf8f0616dee789d3", "8abce8a944c041b980edb46b327b7d7e"]} colab_type="code" id="d4Iatz7WKR-l" outputId="7332db8f-3bc7-4899-c4aa-13aac6b8af25"
plot_model(lightgbm)

# + colab={"base_uri": "https://localhost:8080/", "height": 378} colab_type="code" id="0Hv6iMsZKVg0" outputId="c3003fd8-76c8-4ea3-8228-6657fdd0027d"
plot_model(lightgbm, plot = 'error')

# + colab={"base_uri": "https://localhost:8080/", "height": 349} colab_type="code" id="olNQjb0XKWTh" outputId="4ef70395-74bc-46fc-b33d-1ebf8a45cda3"
plot_model(tuned_lightgbm, plot='feature') # looks like COVID-19 has played a huge role in sales

# + colab={"base_uri": "https://localhost:8080/", "height": 80} colab_type="code" id="b_Fd_xJ1Lgs5" outputId="c5fdaee8-a602-4e7d-b445-22323cbc4cfa"
predict_model(tuned_lightgbm);

# + colab={} colab_type="code" id="JV_UhNbfLifp"
final_lightgbm = finalize_model(tuned_lightgbm)

# + colab={"base_uri": "https://localhost:8080/", "height": 119} colab_type="code" id="URUPWTwkLoRs" outputId="abf319c4-ba69-43e2-aba0-f52d8e30da99"
#Final Light Gradient Boosting Machine parameters for deployment
print(final_lightgbm)

# + colab={"base_uri": "https://localhost:8080/", "height": 80} colab_type="code" id="s8eWLjcrLuAY" outputId="d1dfcc3c-f0e4-4766-b4b2-6cb37a9be06a"
predict_model(final_lightgbm);

# + colab={} colab_type="code" id="wUpeBVkIL5zF"
unseen_predictions = predict_model(final_lightgbm, data=test)
unseen_predictions.head()
unseen_predictions.loc[unseen_predictions['Label'] < 0, 'Label'] = 0 #removing any negative values


# + colab={} colab_type="code" id="SzVa5sgyO3Ao"
def plot_series(time, series,i, format="-", start=0, end=None):
    #plt.figure(figsize=(20,10))
    plt.plot(time[start:end], series[start:end], format,label=i)
    plt.xlabel("Date")
    plt.ylabel("Sales (Dollar)")
    plt.legend()


# + colab={"base_uri": "https://localhost:8080/", "height": 606} colab_type="code" id="EiznLosEO4L3" outputId="fddb144d-cd6f-40a2-dd60-f1b59820c9e5"
plt.figure(figsize=(20,10))
plot_series(test.index, test['sale_dollars'],"True")
#plot_series(train['ds'],train['y'])
plot_series(test.index, unseen_predictions['Label'],"Baseline")


# + [markdown] colab_type="text" id="TI864pAXCRqI"
# Introducing a new metric, SMAPE - this works really well when there are a lot of 0's in the data - like this one. Please note, 0 is not a missing value

# + colab={} colab_type="code" id="ofNoqfhAWlsj"
def calc_smape(y_hat, y):
        return 100/len(y) * np.sum(2 * np.abs(y_hat - y) / (np.abs(y) + np.abs(y_hat)))



# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="1innmkebCPwn" outputId="d66e0cbc-6f0c-432f-9d91-a2801ea4557d"
calc_smape(test['sale_dollars'].values,unseen_predictions['Label'].values)

# + [markdown] colab_type="text" id="hcrc0OUdCezq"
# We will consider 78.3 as our baseline SMAPE 

# + [markdown] colab_type="text" id="Ucb1692uCnPa"
# ## Blending Models
# We will now create a blend model using four algorithms, huber, random forest, xgboost and lightgbm

# + colab={} colab_type="code" id="auWCWPD1RJOV"
#huber = create_model('huber', verbose = False)
rf = create_model('rf', verbose = False)
lightgbm = create_model('lightgbm', verbose = False)
xgb = create_model('xgboost',verbose=False)


# + colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["d895c9fc343641e78aa92389880f7df5", "3dbefe6edba64e1896335a8e743b6c7f", "e51c625ae253409dbb5e4d001a8f2267", "ef5a88c7e4ef4c57a61865b6ac7996a8", "6bd7926964404537beac298835857861", "134db3d23e6e4eb8b84e9e87db30b6b9", "97ed6d6cb2bf481ab396e214f30a1635", "fd23b88b2f4149d2adccd0693c9f7d1a", "6b45ba04086f4a208f17db9c96277fdf", "1b3083c1b17a4a3cbbb0202142b8e593", "f5e95438c1d1498b8590421fe26065a6", "b3d482acf10e4aa380970a4d77cb3a52"]} colab_type="code" id="1Wuh06ZZc5x6" outputId="dd7fa321-9133-4c75-9b4d-051cbba52930"
tuned_rf = tune_model(rf)
tuned_huber = tune_model(huber)
tuned_lightgbm = tune_model(lightgbm)
tuned_xgb = tune_model(xgb)

# + colab={"base_uri": "https://localhost:8080/", "height": 376} colab_type="code" id="4m9PKmv1gieh" outputId="310b5295-464a-4d46-a74b-f022bdfe0551"
plot_model(tuned_huber)

# + [markdown] colab_type="text" id="SAwngrt3Dz5m"
# The below script will just blend all the four models in to one - the time savings are phenomenal

# + colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["b55cc85d024843eaa9bb5c2851e5e5b0", "0222b1b298c047e389f7475901bad635", "00a73ea3bb85411c979a72ff1ef88855"]} colab_type="code" id="tkvnGBfdTrGr" outputId="06364d18-6a9f-4874-c63a-76c5c6ad1552"
blend_specific = blend_models(estimator_list = [tuned_rf,tuned_lightgbm,tuned_xgb,tuned_huber])

# + colab={"base_uri": "https://localhost:8080/", "height": 80} colab_type="code" id="6gjtmrgjUGSE" outputId="ad925eb1-bf42-4f3d-d662-907c96e59c99"
predict_model(blend_specific);

# + colab={} colab_type="code" id="fF1esVyJUExv"
final_model = finalize_model(blend_specific)

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="exxFJKhDUGBv" outputId="ed7dc17a-a7a0-43a4-f184-7051746b37ff"
unseen_predictions_2 = predict_model(final_model, data=test, round=0)
unseen_predictions_2.loc[unseen_predictions_2['Label'] < 0, 'Label'] = 0
unseen_predictions_2.head()

# + colab={"base_uri": "https://localhost:8080/", "height": 334} colab_type="code" id="BMYv_OMwUFhN" outputId="b83d6493-f9e8-44d3-c254-e10da4a181ff"
plt.figure(figsize=(20,5))
plot_series(test.index, test['sale_dollars'],"True")
plot_series(test.index, unseen_predictions_2['Label'],'Blend')


# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="sru6fAMrWnSN" outputId="363bc012-0f30-46b9-e18e-6e165063948d"
calc_smape(test['sale_dollars'].values,unseen_predictions_2['Label'].values)

# + [markdown] colab_type="text" id="ebE92ZFGFuEL"
# The blend model is a major improvment over the baseline model.

# + [markdown] colab_type="text" id="-T5E2_VQF1Zt"
# ## Stacking
# Let's try one more technique, stacking and see if it improves our results

# + colab={} colab_type="code" id="NHslO6ficCto"
stack_1 = stack_models([tuned_rf,tuned_xgb, tuned_lightgbm])
predict_model(stack_1);
final_stack_1 = finalize_model(stack_1)
unseen_predictions_3 = predict_model(final_stack_1, data=test, round=0)



# + colab={"base_uri": "https://localhost:8080/", "height": 173} colab_type="code" id="kfrbaM4OHPEc" outputId="6611e55e-ef80-4a1e-be0d-0c4465b4cf16"
unseen_predictions_3.loc[unseen_predictions_3['Label'] < 0, 'Label'] = 0
unseen_predictions_3.head(4)

# + colab={"base_uri": "https://localhost:8080/", "height": 34} colab_type="code" id="fX5cIpwrcjkh" outputId="d0dffad4-9c12-4f5c-f308-b1d07af27183"
calc_smape(test['sale_dollars'].values,unseen_predictions_3['Label'].values)

# + [markdown] colab_type="text" id="v-3MxDSZGcS2"
# Stacking definitely did not improve the model

# + colab={"base_uri": "https://localhost:8080/", "height": 334} colab_type="code" id="q6_iG787HAhY" outputId="4cbe513f-af9d-4dac-9ca4-45dc180d364d"
plt.figure(figsize=(20,5))
plot_series(test.index, test['sale_dollars'],"True")
plot_series(test.index, unseen_predictions['Label'],'Baseline')
plot_series(test.index, unseen_predictions_2['Label'],'Blend')
plot_series(test.index, unseen_predictions_3['Label'],'Stacking')

# + [markdown] colab_type="text" id="TKOmyM1cGgK6"
# #Next Steps
# The model isn't complete as yet - we can always go back to create a combination of new models + features 

# + colab={} colab_type="code" id="cnZDDXFkkxV6"

