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

# + [markdown] colab_type="text" id="Y57RMM1LEQmR"
# #  <span style="color:orange">Binary Classification  Tutorial (CLF101) - Level Beginner</span>

# + id="3sXuMNuqG6PG" colab_type="code" colab={}
from pycaret.classification import *

# + colab_type="code" id="lUvE187JEQm3" colab={"base_uri": "https://localhost:8080/", "height": 224} outputId="884aba35-318a-4a9d-c002-8f89624b7e51"
from pycaret.datasets import get_data
dataset = get_data('credit')

# + colab_type="code" id="hXmaL1xFEQnj" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="8279a2f7-7be2-4bbf-862b-6227325f685d"
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + id="wGCZvn1K0M9p" colab_type="code" colab={}
exp_clf101 = setup(data = data, target = 'default', session_id=123) 

# + colab_type="code" id="FGCoUiQpEQpz" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["cdbf230f08134c84ad015ebf697c5a37", "3376089bfa864982a9c2007688faa30e", "70ed6f091c554c9eb98de199f1a24681"]} outputId="a8d60f1a-984d-4105-faa4-ec063367423a"

rf = create_model('rf')

# + colab_type="code" id="gmaIfnBMEQrE" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["c1d721acda694b2c8e64bd114a5c2525", "3a81f2aa0e494d3b83233cff789afb10", "977afd7dc03440b4bf967c87ac7a6a03"]} outputId="fc41b7bd-af08-4658-eb0b-8ecfc124bfa6"
tuned_rf = tune_model(rf)

# + colab_type="code" id="nwaZk6oTEQsi" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="c60182d4-b44e-4853-fe5f-02701c885d5e"
predict_model(tuned_rf);

# + [markdown] colab_type="text" id="r79BGjIfEQs1"
# # 12.0 Finalize Model for Deployment

# + colab_type="code" id="_--tO4KGEQs-" colab={}
final_rf = finalize_model(tuned_rf)

# + colab_type="code" id="U9W6kXsSEQtQ" colab={"base_uri": "https://localhost:8080/", "height": 161} outputId="eb354276-8fe0-476e-e1b5-d5649a4f22c8"
#Final Random Forest model parameters for deployment
print(final_rf)

# + colab_type="code" id="NJDk3I-EEQtg" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="64d82cb1-c8ff-4440-c977-17e280920e23"
predict_model(final_rf);

# + [markdown] id="dWU2Dmdx2UNZ" colab_type="text"
# # 13.0 Deploy Model on Microsoft Azure
#
# This is the code to deploy model on Microsft azure using `pycaret` functionalities.

# + id="PtdFIPJJ0zHX" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 287} outputId="6d118b7d-6c29-4f1a-8e4a-3a767dae6f95"
# ! pip install azure-storage-blob



# + id="ImFnwpb52iDl" colab_type="code" colab={}
## Enter connection string when running in google colab
connect_str = '' #@param {type:"string"}
print(connect_str)

# + id="4FolddlO2iTK" colab_type="code" colab={}
# #! export AZURE_STORAGE_CONNECTION_STRING=connect_str

# + id="q_MZPZ4271g3" colab_type="code" colab={}
os.environ['AZURE_STORAGE_CONNECTION_STRING']= connect_str

# + id="wz0YIfLb6iVK" colab_type="code" colab={}
# ! echo $AZURE_STORAGE_CONNECTION_STRING

# + id="cUOqSvi63m01" colab_type="code" colab={}
os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# + id="H3C-nMpF2iZg" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 89} outputId="f6c6dbdb-d0c6-400a-d43c-8ee0342edfa1"
authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
deploy_model(final_rf, model_name, authentication, platform = 'azure')

# + id="iuBz98UT2icD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 125} outputId="dddd993d-510f-4ffa-bf89-bfff8ad5f279"
authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
model_azure = load_model(model_name, 
               platform = 'azure', 
               authentication = authentication,
               verbose=True)

# + id="aiP_EiLm2iWk" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 125} outputId="8243e67b-79a0-4801-a38e-dfd3fbbf4c82"
authentication = {'container': 'pycaret-cls-101'}
model_name = 'rf-clf-101'
unseen_predictions = predict_model(model_name, data=data_unseen, platform='azure', authentication=authentication, verbose=True)

# + id="UkX3mtAD2iJH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 439} outputId="98be3bdd-0064-4da4-e488-234668cd5f40"
unseen_predictions

# + id="2CRqugcz2h5a" colab_type="code" colab={}


# + [markdown] id="0ZxYxszDBqJh" colab_type="text"
# # 13.0 Deploy Model on Google Cloud

# + [markdown] id="N5qy_gsfB1rA" colab_type="text"
# After the model is finalised and you are happy with the model, you can deploy the model on your cloud of choice. In this section, we deploy the model on the google cloud platform. 

# + id="2eJdBC3EClnW" colab_type="code" colab={}
from google.colab import auth
auth.authenticate_user()

# + id="9L31JPblEPG6" colab_type="code" colab={}
# ! pip install awscli

# + id="i8xWrcliQCz1" colab_type="code" colab={}
# GCP project name, Change the name based on your own GCP project.
CLOUD_PROJECT = 'gcpessentials-rz' # GCP project name
bucket_name = 'pycaret-clf101-test1' # bucket name for storage of your model
BUCKET = 'gs://' + CLOUD_PROJECT + '-{}'.format(bucket_name)
# Set the gcloud consol to $CLOUD_PROJECT Environment Variable for your Desired Project)
# !gcloud config set project $CLOUD_PROJECT

# + id="fq7-Su1iQuHl" colab_type="code" colab={}
authentication = {'project': CLOUD_PROJECT, 'bucket' : bucket_name}
model_name = 'rf-clf'
deploy_model(final_rf, model_name, authentication, platform = 'gcp')

# + id="CN0CkUXKRAlc" colab_type="code" colab={}
authentication = {'project': CLOUD_PROJECT, 'bucket' : bucket_name}
model_name = 'rf-clf'
model_gcp = load_model(model_name, 
               platform = 'gcp', 
               authentication = authentication,
               verbose=True)

# + id="hlY68MyNbZ8d" colab_type="code" colab={}
estimator_ = load_model(model_name, platform='gcp',
                                   authentication=authentication,
                                   verbose=True)

# + id="bIMlREBHXTtF" colab_type="code" colab={}
authentication = {'project': CLOUD_PROJECT, 'bucket' : bucket_name}
model_name = 'rf-clf'
unseen_predictions = predict_model(model_name, data=data_unseen, platform='gcp', authentication=authentication, verbose=True)

# + id="CFxn0KJ_ebGz" colab_type="code" colab={}
unseen_predictions

# + id="8lEM1JyDcB_3" colab_type="code" colab={}
authentication

# + id="SGY9ZiecTC1X" colab_type="code" colab={}
import inspect as i
import sys
sys.stdout.write(i.getsource(predict_model))


# + [markdown] colab_type="text" id="hUzc6tXNEQtr"
# # 13.0 Predict on unseen data

# + [markdown] colab_type="text" id="dx5vXjChEQtt"
# The `predict_model()` function is also used to predict on the unseen dataset. The only difference from section 11 above is that this time we will pass the `data_unseen` parameter. `data_unseen` is the variable created at the beginning of the tutorial and contains 5% (1200 samples) of the original dataset which was never exposed to PyCaret. (see section 5 for explanation)

# + colab_type="code" id="0y5KWLC6EQtx" colab={}
unseen_predictions = predict_model(final_rf, data=data_unseen)
unseen_predictions.head()

# + [markdown] colab_type="text" id="oPYmVpugEQt5"
# The `Label` and `Score` columns are added onto the `data_unseen` set. Label is the prediction and score is the probability of the prediction. Notice that predicted results are concatenated to the original dataset while all the transformations are automatically performed in the background.

# + [markdown] colab_type="text" id="L__po3sUEQt7"
# # 14.0 Saving the model

# + [markdown] colab_type="text" id="1sQPT7jrEQt-"
# We have now finished the experiment by finalizing the `tuned_rf` model which is now stored in `final_rf` variable. We have also used the model stored in `final_rf` to predict `data_unseen`. This brings us to the end of our experiment, but one question is still to be asked: What happens when you have more new data to predict? Do you have to go through the entire experiment again? The answer is no, PyCaret's inbuilt function `save_model()` allows you to save the model along with entire transformation pipeline for later use.

# + colab_type="code" id="ln1YWIXTEQuA" colab={}
save_model(final_rf,'Final RF Model 08Feb2020')

# + [markdown] colab_type="text" id="WE6f48AYEQuR"
# (TIP : It's always good to use date in the filename when saving models, it's good for version control.)

# + [markdown] colab_type="text" id="Z8OBesfkEQuU"
# # 15.0 Loading the saved model

# + [markdown] colab_type="text" id="V2K_WLaaEQuW"
# To load a saved model at a future date in the same or an alternative environment, we would use PyCaret's `load_model()` function and then easily apply the saved model on new unseen data for prediction.

# + colab_type="code" id="Siw_2EIUEQub" colab={}
saved_final_rf = load_model('Final RF Model 08Feb2020')

# + [markdown] colab_type="text" id="1zyi6-Q-EQuq"
# Once the model is loaded in the environment, you can simply use it to predict on any new data using the same `predict_model()` function. Below we have applied the loaded model to predict the same `data_unseen` that we used in section 13 above.

# + colab_type="code" id="HMPO1ka9EQut" colab={}
new_prediction = predict_model(saved_final_rf, data=data_unseen)

# + colab_type="code" id="7wyDQQSzEQu8" colab={}
new_prediction.head()

# + [markdown] colab_type="text" id="bf8I1uqcEQvD"
# Notice that the results of `unseen_predictions` and `new_prediction` are identical.

# + [markdown] colab_type="text" id="_HeOs8BhEQvF"
# # 16.0 Wrap-up / Next Steps?

# + [markdown] colab_type="text" id="VqG1NnwXEQvK"
# This tutorial has covered the entire machine learning pipeline from data ingestion, pre-processing, training the model, hyperparameter tuning, prediction and saving the model for later use. We have completed all of these steps in less than 10 commands which are naturally constructed and very intuitive to remember such as `create_model()`, `tune_model()`, `compare_models()`. Re-creating the entire experiment without PyCaret would have taken well over 100 lines of code in most libraries.
#
# We have only covered the basics of `pycaret.classification`. In following tutorials we will go deeper into advanced pre-processing, ensembling, generalized stacking and other techniques that allow you to fully customize your machine learning pipeline and are must know for any data scientist.
#
# See you at the next tutorial. Follow the link to __[Binary Classification Tutorial (CLF102) - Intermediate Level](https://github.com/pycaret/pycaret/blob/master/Tutorials/Binary%20Classification%20Tutorial%20Level%20Intermediate%20-%20CLF102.ipynb)__
