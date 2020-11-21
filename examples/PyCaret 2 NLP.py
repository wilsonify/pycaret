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

# # PyCaret 2 NLP Example
# This notebook is created using PyCaret 2.0. Last updated : 31-07-2020

# check version
from pycaret.utils import version
version()

# # 1. Loading Dataset

from pycaret.datasets import get_data
data = get_data('kiva')

# # 2. Initialize Setup

from pycaret.nlp import *
nlp1 = setup(data, target = 'en', session_id=123, log_experiment=True, log_plots = True, experiment_name='kiva1')

# # 3. Create Model

models()

lda = create_model('lda')

nmf = create_model('nmf', num_topics = 6)

# # 4. Assign Labels

lda_results = assign_model(lda)
lda_results.head()

# # 5. Analyze Model

plot_model(lda)

plot_model(lda, plot = 'bigram')

plot_model(lda, plot = 'tsne')

evaluate_model(lda)

# # 6. MLFlow UI

# !mlflow ui

# # End
# Thank you. For more information / tutorials on PyCaret, please visit https://www.pycaret.org
