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

# # PyCaret 2 Anomaly Example
# This notebook is created using PyCaret 2.0. Last updated : 31-07-2020

# check version
from pycaret.utils import version
version()

# # 1. Loading Dataset

from pycaret.datasets import get_data
data = get_data('anomaly')

# # 2. Initialize Setup

from pycaret.anomaly import *
ano1 = setup(data, session_id=123, log_experiment=True, experiment_name='anomaly1')

# # 3. Create Model

models()

iforest = create_model('iforest')

knn = create_model('knn', fraction = 0.1)

# # 4. Assign Labels

iforest_results = assign_model(iforest)
iforest_results.head()

# # 5. Analyze Model

plot_model(iforest)

plot_model(iforest, plot = 'umap')

# # 6. Predict Model

pred_new = predict_model(iforest, data=data)
pred_new.head()

# # 7. Save / Load Model

save_model(iforest, model_name='iforest')

loaded_iforest = load_model('iforest')
print(loaded_iforest)

from sklearn import set_config
set_config(display='diagram')
loaded_iforest[0]

from sklearn import set_config
set_config(display='text')

# # 8. Deploy Model

deploy_model(iforest, model_name = 'iforest-aws', authentication = {'bucket' : 'pycaret-test'})

# # 9. Get Config / Set Config

X = get_config('X')
X.head()

get_config('seed')

from pycaret.anomaly import set_config
set_config('seed', 999)

get_config('seed')

# # 10. MLFlow UI

# !mlflow ui

# # End
# Thank you. For more information / tutorials on PyCaret, please visit https://www.pycaret.org
