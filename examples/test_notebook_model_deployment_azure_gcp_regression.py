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

# + [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/github/amjadraza/pycaret/blob/feature%2Fgcp_azure_np_docs/examples/test_notebook_model_deployment_azure_gcp_regression.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="LrRweUsz9bSl" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="572e1550-2c0b-4a99-ed67-1d6de4f40a52"
# ! pip uninstall pycaret
# !pip install git+https://github.com/amjadraza/pycaret.git@feature/gcp_azure_np_docs

# + colab_type="code" id="lUvE187JEQm3" colab={"base_uri": "https://localhost:8080/", "height": 204} outputId="e6083dca-71b1-40b6-fc25-3256960e4dfb"
from pycaret.datasets import get_data
dataset = get_data('diamond')

# + colab_type="code" id="hXmaL1xFEQnj" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="da4af24e-212b-4c5f-a4ba-42dbbd7a2953"
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + id="3sXuMNuqG6PG" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 958, "referenced_widgets": ["efaff962f0a94c8d8f5d06311e023506", "c2e2060613d444078b48b60e30a112e7", "09690916f8634c01b8b289d83119e178", "dbb0c8ed8d2f4b37a492d777d1cfcd48", "5db0cd1626b044b6ada0e461e2d7c94b", "8a5e8367e06b49dcbe1550b7ce9a8380"]} outputId="0bd58ff5-896d-4656-96c4-535089378636"
from pycaret.regression import *
exp_reg101 = setup(data = data, target = 'Price', session_id=123)

# + colab_type="code" id="FGCoUiQpEQpz" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["f82b0cd6d54b427ea21cf2a5b2958c21", "0f57ccbc1ce543ad9b7d837c7fe03595", "2dae5a1a055343ea88a268f14cc53a49"]} outputId="79640529-8cdd-4875-b6c7-ce0544252663"
lightgbm = create_model('lightgbm')

# + colab_type="code" id="gmaIfnBMEQrE" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["1de1da0d08c64148935f305cafdeaadc", "577eb34fbd6f4bdea66489337463c354", "f9312a23e9af43e183ebd777b133a2a9"]} outputId="bc3b8389-0387-44d4-9fd9-6d71903e2230"
tuned_lightgbm = tune_model(lightgbm)

# + colab_type="code" id="nwaZk6oTEQsi" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="594ed7b7-0b2d-4a90-ab31-a0bace294024"
predict_model(tuned_lightgbm );

# + [markdown] colab_type="text" id="r79BGjIfEQs1"
# # 12.0 Finalize Model for Deployment

# + colab_type="code" id="_--tO4KGEQs-" colab={}
final_lightgbm  = finalize_model(tuned_lightgbm )

# + colab_type="code" id="U9W6kXsSEQtQ" colab={"base_uri": "https://localhost:8080/", "height": 125} outputId="4bfc1789-f50e-4dc5-aa76-8ec80e7fe6b4"
print(final_lightgbm)

# + colab_type="code" id="NJDk3I-EEQtg" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="18b12734-b0fb-4270-daca-c177d4dbb16c"
predict_model(final_lightgbm);

# + [markdown] id="dWU2Dmdx2UNZ" colab_type="text"
# # 13.0 Deploy Model on Microsoft Azure
#
# This is the code to deploy model on Microsft azure using `pycaret` functionalities.

# + id="PtdFIPJJ0zHX" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 485} outputId="58dee8d4-02f9-4018-c3ba-3dd3209628ff"
# # ! pip install azure-storage-blob
# ! pip install awscli



# + id="ImFnwpb52iDl" colab_type="code" colab={}
## Enter connection string when running in google colab
connect_str = '' #@param {type:"string"}
print(connect_str)

# + id="4FolddlO2iTK" colab_type="code" colab={}
# #! export AZURE_STORAGE_CONNECTION_STRING=connect_str

# + id="q_MZPZ4271g3" colab_type="code" colab={}
import os
os.environ['AZURE_STORAGE_CONNECTION_STRING']= connect_str

# + id="wz0YIfLb6iVK" colab_type="code" colab={}
# ! echo $AZURE_STORAGE_CONNECTION_STRING

# + id="cUOqSvi63m01" colab_type="code" colab={}
os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# + id="H3C-nMpF2iZg" colab_type="code" colab={}
authentication = {'container': 'pycaret-reg-1011'}
model_name = 'lightgbm-reg-101'
deploy_model(final_lightgbm, model_name, authentication, platform = 'azure')

# + id="iuBz98UT2icD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 107} outputId="946b34d4-aadf-470b-bac5-5fd49d8f2198"
authentication = {'container': 'pycaret-reg-1011'}
model_name = 'lightgbm-reg-101'
model_azure = load_model(model_name, 
               platform = 'azure', 
               authentication = authentication,
               verbose=True)

# + id="aiP_EiLm2iWk" colab_type="code" colab={}

unseen_predictions = predict_model(model_azure, data=data_unseen, verbose=True)

# + id="G9s2LdGIbIlV" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 519} outputId="1a3ce935-f82c-40ff-b7f9-1fb816e7e090"
predict_model(model_azure)

# + id="UkX3mtAD2iJH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="a23a5132-24e3-4a65-81cf-8f791e4b57a0"
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

# + id="i8xWrcliQCz1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="77c554fd-d401-4186-c755-741465ba1806"
# GCP project name, Change the name based on your own GCP project.
CLOUD_PROJECT = 'gcpessentials-rz' # GCP project name
bucket_name = 'pycaret-reg101-test1' # bucket name for storage of your model
BUCKET = 'gs://' + CLOUD_PROJECT + '-{}'.format(bucket_name)
# Set the gcloud consol to $CLOUD_PROJECT Environment Variable for your Desired Project)
# !gcloud config set project $CLOUD_PROJECT

# + id="fq7-Su1iQuHl" colab_type="code" colab={}
authentication = {'project': CLOUD_PROJECT, 'bucket' : bucket_name}
model_name = 'lightgbm-reg'
deploy_model(final_lightgbm, model_name, authentication, platform = 'gcp')

# + id="CN0CkUXKRAlc" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="9abf26f2-377f-4327-89af-07ffbab14bc5"
authentication = {'project': CLOUD_PROJECT, 'bucket' : bucket_name}
model_name = 'lightgbm-reg'
model_gcp = load_model(model_name, 
               platform = 'gcp', 
               authentication = authentication,
               verbose=True)

# + id="bIMlREBHXTtF" colab_type="code" colab={}

unseen_predictions = predict_model(model_gcp, data=data_unseen, verbose=True)

# + id="CFxn0KJ_ebGz" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 419} outputId="cffebb43-47cb-4482-fb6d-777d65ac1e5d"
unseen_predictions

# + id="tzpXE4Jmbull" colab_type="code" colab={}

