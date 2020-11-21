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

# # Implementing PyCaret 2.0 on Wine Quality Dataset (Highly Imbalance Classes Problem)

# ### **Since PyCaret 2.0 is recently launched, i plan to work with it and see if this is as amazing as it is hyped. To be honest i found it wonderful, they way it has reduced the work for data scientists and provide low code model to automate most of the repetition work is marvelous. Let start and see for yourself how it can save precious time for users.**

# ### **I've selected Wine Quality dataset due to its highly imbalance classes which is a key problem in dataset for incuring bias.**

# + _uuid="d629ff2d2480ee46fbb7e2d37f6b5fab8052498a" _cell_guid="79c7e3d0-c299-4dcb-8224-4455121ee9b0"
#important libraries
import numpy as np 
import pandas as pd
# -

# **importing wine quality data from UCI ML repo via wget**

# **wget: helps to retrieves content from web servers**

# ## 1. Importing Dataset and PyCaret 2.0

# !wget --no-check-certificate \
#     https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv

df1 = pd.read_csv('/kaggle/working/winequality-red.csv', sep=";")

df1.head()

df1.describe()

# **you can see for yourself how imbalance the classes are, target quality for class 4,8,3 are negligible as compared to 5,6,7. This will surely impact the model performace on minority classes and introduce bias**

df1.quality.value_counts()

# **installing the latest version of PyCaret**

pip install pycaret==2.0

# **Since Problem is classification, you can read about other options here: [Modules](https://pycaret.org/modules/)**

from pycaret.classification import *

# **The use case is simple you have to setup a preprocessing pipeline with 'setup' module, select the target and name your experiment.**
#
# **you can specify a session_id, if you didn't by default a random seed is generated and returned in the Information grid. The unique number is then distributed as a seed in all functions used during the experiment. This can be used for later reproducibility of the entire experiment.**

# ## 2. Using Default Setup of PyCaret 2.0

session_1 = setup(df1, target = 'quality', session_id=123, log_experiment=False, experiment_name='wine_q1')

# **you can complete details about all functions here [1](https://pycaret.org/classification/)**

# **compare_model(): This function train all the models available in the model library and scores them.**
#
# **I blacklisted 'catboost' as it was taking much time in k-folds, you can test if it helps in accuracy and other metrics**

best_model = compare_models(blacklist=['catboost'])

# **models() provide list of models in library and their id that can be used in functions**

models()

# ### **creating 4 best model from default setup output and tuning them to see improvements**

#creating random forest model
rf = create_model('rf')

#creating Extra Trees Classifier
et = create_model('et')

#light GBM Model
lgbm = create_model('lightgbm')

#creating XGB model
xgboost = create_model('xgboost')

# **You can witness how easy is to create model and CV it**
#
# **Similarly, you can tune your model with best hyper params with just single line of code, isn't it amazing**

#Hyper params tuning via tune_model
tuned_rf = tune_model(rf)

# **And you can see the params of your best model as well**

tuned_rf

tuned_et = tune_model(et)

# ### **Plotting ROC for initial/basic model**

plot_model(rf)

# ### **Plotting ROC for hyper parameters tuned model**

plot_model(tuned_rf)

# ### **This is an interesting way of seeing how our model create boundry and how it fits the space of features, below you can see our model is overfit**

plot_model(rf, plot = 'boundary')

# ### **Do you see the difference in boundries of models (normal and tuned)**

plot_model(tuned_rf, plot = 'boundary')

# ## 3. Oversampling to balance the classes

from imblearn.over_sampling import *
adasyn1 = ADASYN(sampling_strategy='minority')

# ### **Here i have worked and tweaked some of the functions of the pipeline and try to balance the imbalance classes**

Session_2 = setup(df1, target = 'quality', session_id=177, log_experiment=False, 
                  experiment_name='wine_q2', normalize=True, normalize_method='zscore', 
                  transformation=True, transformation_method = 'quantile', fix_imbalance=True,
                  fix_imbalance_method= adasyn1)

# **You can several of the func are now shahed green because i will be envolving them in pipeline**

best_model1 = compare_models(blacklist=['catboost'])

xgboost_1 = create_model('xgboost')

et_1 = create_model('et')

lgbm_1 = create_model('lightgbm')

gbc_1 = create_model('gbc')

tuned_xgboost1 = tune_model(xgboost_1)

tuned_lgbm1 = tune_model(lgbm_1)

tuned_et1 = tune_model(et_1)

tuned_gbc1 = tune_model(gbc_1)

# ### **This function creates a Soft Voting(Majority Rule classifier) for the selected estimators in the model**

blend_soft = blend_models(estimator_list = [tuned_lgbm1, tuned_xgboost1, tuned_gbc1, tuned_et1], method = 'soft')

# **Check the ROC Curves w.r.t to default setup and check improvements**

plot_model(blend_soft)

stacked_lgbm = stack_models(estimator_list = [tuned_lgbm1, tuned_xgboost1, tuned_gbc1, tuned_et1],
                           meta_model=lgbm_1)

# ### **Now Comparing best tuned model with blended model**

plot_model(lgbm_1, plot = 'confusion_matrix')

plot_model(blend_soft, plot = 'confusion_matrix')

plot_model(blend_soft, plot = 'boundary')

plot_model(lgbm_1, plot = 'boundary')

plot_model(blend_soft, plot = 'pr')

plot_model(lgbm_1, plot = 'pr')

plot_model(blend_soft, plot = 'class_report')

plot_model(lgbm_1, plot = 'class_report')

plot_model(blend_soft, plot = 'error')

plot_model(lgbm_1, plot = 'error')

plot_model(lgbm_1, plot = 'dimension')

# ### **See, how we can play and tune our models within just few lines of code (given the knowlegde of models, math and paramters working) this will help automate most of the hectic work of data scientists.**

# **Thanks for checking this notebook, you can contact me here for queries or at [LinkedIn](https://www.linkedin.com/in/muhammad-saad-31740060/)**
