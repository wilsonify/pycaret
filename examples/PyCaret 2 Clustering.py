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

# # PyCaret 2 Clustering Example
# This notebook is created using PyCaret 2.0. Last updated : 31-07-2020

# check version
from pycaret.utils import version
version()

# # 1. Loading Dataset

from pycaret.datasets import get_data
data = get_data('public_health')

# # 2. Initialize Setup

from pycaret.clustering import *
clu1 = setup(data, ignore_features = ['Country Name'], session_id=123, log_experiment=True, log_plots = True, 
             experiment_name='health1')

# # 3. Create Model

models()

kmeans = create_model('kmeans', num_clusters = 4)

kmodes = create_model('kmodes', num_clusters = 4)

# # 4. Assign Labels

kmeans_results = assign_model(kmeans)
kmeans_results.head()

# # 5. Analyze Model

plot_model(kmeans)

plot_model(kmeans, feature = 'Country Name', label=True)

plot_model(kmeans, plot = 'tsne')

plot_model(kmeans, plot = 'elbow')

plot_model(kmeans, plot = 'silhouette')

plot_model(kmeans, plot = 'distance')

plot_model(kmeans, plot = 'distribution')

# # 6. Predict Model

pred_new = predict_model(kmeans, data=data)
pred_new.head()

# # 7. Save / Load Model

save_model(kmeans, model_name='kmeans')

loaded_kmeans = load_model('kmeans')
print(loaded_kmeans)

from sklearn import set_config
set_config(display='diagram')
loaded_kmeans[0]

from sklearn import set_config
set_config(display='text')

# # 8. Deploy Model

deploy_model(kmeans, model_name = 'kmeans-aws', authentication = {'bucket' : 'pycaret-test'})

# # 9. Get Config / Set Config

X = get_config('X')
X.head()

get_config('seed')

from pycaret.clustering import set_config
set_config('seed', 999)

get_config('seed')

# # 10. MLFlow UI

# !mlflow ui

# # End
# Thank you. For more information / tutorials on PyCaret, please visit https://www.pycaret.org
