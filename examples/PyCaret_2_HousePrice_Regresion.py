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

# + [markdown] id="I76BhvxwrMfn" colab_type="text"
# # PyCaret 2 House Price Prediction Example

# + [markdown] id="wSSQFNvlrMfp" colab_type="text"
# This notebook is created using PyCaret 2.0. Last updated : 04-08-2020

# + [markdown] colab_type="text" id="gobl5HA-juwP"
# House Price Prediction data set from Kaggle https://www.kaggle.com/c/house-prices-advanced-regression-techniques <br>
# Train Dataset consists of 1460 Samples with 81 features including the SalePrice<br>
# Test Dataset consists of 1459 Samples wit 80 features

# + colab_type="code" id="ICXiqtC_TacA" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="49a1f1f6-c4e7-499a-eb34-4d19bceb4562"
# Mount Google Drive 
# Skip this step if using on local hardware 
from google.colab import drive
drive.mount('/content/gdrive')

# + colab_type="code" id="pD-STV05yakD" colab={}
# Works with pycaret and pycaret 2
# #!pip install pycaret==2.0
from pycaret.regression import *
import pandas as pd

# + id="y0qd3_m7rMfu" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="eff3c677-6bf6-4e21-de39-2c1e901254ce"
# check version
from pycaret.utils import version
version()

# + colab_type="code" id="IVl3sVtl32im" colab={"base_uri": "https://localhost:8080/", "height": 34} outputId="5c427692-f3ce-400f-a5ac-ae916ad6d11c"
# Chane path as per your file structure
# Remove root_path if using local hardware

root_path = 'gdrive/My Drive/Colab Notebooks/'

data = pd.read_csv('gdrive/My Drive/Colab Notebooks/HousePrice/train.csv')

test_data = pd.read_csv('gdrive/My Drive/Colab Notebooks/HousePrice/test.csv')

print(data.shape, test_data.shape)

# + colab_type="code" id="kpCdQOa9ZZnH" colab={"base_uri": "https://localhost:8080/", "height": 256} outputId="7faadb2f-8e76-4a34-a85e-e24f285a5113"
data.head()

# + colab_type="code" id="hRT-l3CUYvgW" colab={"base_uri": "https://localhost:8080/", "height": 973, "referenced_widgets": ["84cd4922236448a3a306a3f6e7e43c88", "47375cce9026442183ab428080b82e11", "fefbae2c76574bf7b31541e5f1e022ef", "562caa85cf7d4a2fbf4decac3fc6983b", "30410561071f463ab76e219406f2fa9c", "55e0f67422e244a9842b8dcc917d493d"]} outputId="6834803b-b97a-478b-e92b-e262e6bdf6a6"
# Ignoring features with high null values 

demo = setup(data = data, target = 'SalePrice', 
                   ignore_features = ['Alley','PoolQC','MiscFeature','Fence','FireplaceQu','Utilities'],normalize = True,
                   transformation= True, transformation_method = 'yeo-johnson', 
                   transform_target = True, remove_outliers= True,
                   remove_multicollinearity = True,
                   ignore_low_variance = True, combine_rare_levels = True) 

# + colab_type="code" id="v8tsvaw4aHHp" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["cd5ab1a14de5402985b82653a139d818", "6e28825ce0c1497287c54a26ed64fb9b", "c45d6e58845e4a7e860c7fde14ed94d2"]} outputId="1708f0d0-a5ff-4841-a3f5-634598ee507a"
# Blacklist Theil–Sen Regressor 
# Auto sort on R2 
compare_models(blacklist = ['tr'])

# + colab_type="code" id="wXxDMgpWe_nj" colab={"base_uri": "https://localhost:8080/", "height": 297, "referenced_widgets": ["3c29f19233864b53ba8da01af51d572a", "c59e1582725c43f282f196e76d7f8883", "629dd21d247a4a63a3705add599b59dd", "5ea1974eab714e209d781a04564f38aa", "51b30e1917584733bcc1ad280f657a6f", "e3c23703d9304ea09d7233dadc0b067f", "3d8c5beac0da40168f86729f70acbb79", "08c8a60ba9e94046bd2d98afe7cc741a", "367ad9468f49427ca8475e5301a95ddb"]} outputId="1e60ba0d-cf32-4c8e-cb9b-a084e6a0106d"
# Creating models for the best estimators 
huber = create_model('huber')
bayesian_ridge = create_model('br')
cat_boost = create_model('catboost')

# + colab_type="code" id="RyfTEP0RhzAy" colab={"base_uri": "https://localhost:8080/", "height": 297, "referenced_widgets": ["838a3f746e6a4b6b9f3d05d6c3d506cb", "3a2d5a31c1d1444ab603e67934aa6acc", "11fdc00d65ad4d82950b754c7f5ef6b0", "3e621330c9144f568bdc0eef124afb1a", "ce93bca57c684aa3b46265d0c37eb746", "c485ec5deb47455fad921d447491bb7f", "5ff4ebf9ceed4072bf633bb282423dca", "17633bba196a4c1897ecce8482066c29", "583f1dc1beea4d75b9fe1c1f1de719e1"]} outputId="65c8f299-4e28-4131-8d96-6523ab5edfca"
# Tuning the created models 
huber = tune_model(huber)
bayesian_ridge = tune_model(bayesian_ridge)
cat_boost = tune_model(cat_boost)

# + colab_type="code" id="rjXOLZR6lTRf" colab={"base_uri": "https://localhost:8080/", "height": 297, "referenced_widgets": ["47c66341f17d40de815a82eedaf43dba", "c77287c6a0dd4892b02f8e94f83360d6", "2656c119068c4faa86003a43ddb52453"]} outputId="da06d194-0efd-40e3-9b50-f35120d5e76c"
# Blending models
blender = blend_models(estimator_list = [huber, bayesian_ridge, cat_boost])

# + colab_type="code" id="dT_dnYL1na25" colab={}
# Finaliszing model for predictions 
model = finalize_model(blender)
predictions = predict_model(model, data = test_data)

# + colab_type="code" id="_3lQxyHcoCzH" colab={}
# Generating CSV for Kaggle Submissions 
sub = pd.DataFrame({
        "Id": predictions['Id'],
        "SalePrice": predictions['Label']
    })

sub.to_csv('gdrive/My Drive/Colab Notebooks/HousePrice/submission.csv', index=False)
