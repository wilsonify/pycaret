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

# # PyCaret 2 Association Rule Mining Example
# This notebook is created using PyCaret 2.0. Last updated : 31-07-2020

from pycaret.utils import version
version()

# # 1. Loading dataset

from pycaret.datasets import get_data
data = get_data('france')

# # 2. Init setup

from pycaret.arules import *

s = setup(data, 'InvoiceNo', item_id='Description')

# # 3. Create Model

model1 = create_model()

model1

# # 4. Plot Model

plot_model(model1)

plot_model(model1, plot='3d')


