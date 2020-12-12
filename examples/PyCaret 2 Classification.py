"""
PyCaret 2 Classification Example
This notebook is created using PyCaret 2.0. Last updated : 31-07-2020
"""
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


# check version
from pycaret.utils import version

version()

# # 1. Data Repository

from pycaret.datasets import get_data

index = get_data('index')

data = get_data('juice')

# # 2. Initialize Setup

from pycaret.classification import *

clf1 = setup(data, target='Purchase', session_id=123, log_experiment=True, experiment_name='juice1')

# # 3. Compare Baseline

best_model = compare_models()

# # 4. Create Model

lr = create_model('lr')

dt = create_model('dt')

rf = create_model('rf', fold=5)

models()

models(type='ensemble').index.tolist()

ensembled_models = compare_models(include=models(type='ensemble').index.tolist(), fold=3)

# # 5. Tune Hyperparameters

tuned_lr = tune_model(lr)

tuned_rf = tune_model(rf)

# # 6. Ensemble Model

bagged_dt = ensemble_model(dt)

boosted_dt = ensemble_model(dt, method='Boosting')

# # 7. Blend Models

blender = blend_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], method='soft')

# # 8. Stack Models

stacker = stack_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], meta_model=rf)

# # 9. Analyze Model

plot_model(rf)

plot_model(rf, plot='confusion_matrix')

plot_model(rf, plot='boundary')

plot_model(rf, plot='feature')

plot_model(rf, plot='pr')

plot_model(rf, plot='class_report')

evaluate_model(rf)

# # 10. Interpret Model

catboost = create_model('catboost', cross_validation=False)

interpret_model(catboost)

interpret_model(catboost, plot='correlation')

interpret_model(catboost, plot='reason', observation=12)

# # 11. AutoML()

best = automl(optimize='Recall')
best

# # 12. Predict Model

pred_holdouts = predict_model(lr)
pred_holdouts.head()

new_data = data.copy()
new_data.drop(['Purchase'], axis=1, inplace=True)
predict_new = predict_model(best, data=new_data)
predict_new.head()

# # 13. Save / Load Model

save_model(best, model_name='best-model')

loaded_bestmodel = load_model('best-model')
print(loaded_bestmodel)

from sklearn import set_config

set_config(display='diagram')
loaded_bestmodel[0]

from sklearn import set_config

set_config(display='text')

# # 14. Deploy Model

deploy_model(best, model_name='best-aws', authentication={'bucket': 'pycaret-test'})

# # 15. Get Config / Set Config

X_train = get_config('X_train')
X_train.head()

get_config('seed')

from pycaret.classification import set_config

set_config('seed', 999)

get_config('seed')
