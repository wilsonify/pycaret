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
# <a href="https://colab.research.google.com/github/amjadraza/pycaret/blob/feature%2Fgcp_azure_np_docs/examples/test_notebook_model_deployment_azure_gcp_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] colab_type="text" id="Y57RMM1LEQmR"
# #  <span style="color:orange">Binary Classification  Tutorial (CLF101) - Level Beginner</span>

# + id="LrRweUsz9bSl" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 1000} outputId="5851e7fa-b6d8-409a-8e0d-c63fd4b843f7"
# ! pip uninstall pycaret
# !pip install git+https://github.com/amjadraza/pycaret.git@feature/gcp_azure_np_docs

# + id="3sXuMNuqG6PG" colab_type="code" colab={}
from pycaret.classification import *

# + colab_type="code" id="lUvE187JEQm3" colab={"base_uri": "https://localhost:8080/", "height": 224} outputId="c39d1398-9edf-417f-e888-3590124026f8"
from pycaret.datasets import get_data
dataset = get_data('credit')

# + colab_type="code" id="hXmaL1xFEQnj" colab={"base_uri": "https://localhost:8080/", "height": 53} outputId="00c8ed45-e151-44bf-a5fe-961a1290e253"
data = dataset.sample(frac=0.95, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

# + id="wGCZvn1K0M9p" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 982, "referenced_widgets": ["aa085a97390d4dfbad05ee40637f92ff", "25dd5860e4a2432c8ccedab57e49d416", "070deab125f34d21a52c1a6739c01248", "e6a988efbd0b4ccf86508eb631a53c27", "a5eef38b75284ef894c7890820550315", "9ecd3015a12b498fbdc3a1e7f9ece621"]} outputId="c09bf157-50f1-4b36-cb01-2a0c704dd2d8"
exp_clf101 = setup(data = data, target = 'default', session_id=123) 

# + colab_type="code" id="FGCoUiQpEQpz" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["cde834ccd7a340d7b775f28c6e723f22", "919fb81e19164fa4bba0a89eb4b01cc0", "30796291c40848778ba283225350cc30"]} outputId="8e3b0b24-53af-41fe-9c59-142dcd952cd8"

rf = create_model('rf')

# + colab_type="code" id="gmaIfnBMEQrE" colab={"base_uri": "https://localhost:8080/", "height": 292, "referenced_widgets": ["c2bf7e77cf3442e381aa178f42a621e9", "000c5b9811a942019a2783601f7aa4fd", "cc386b47cecb458d8e2cfb43f6519770"]} outputId="b168276a-bdf5-4097-cb6a-c57a8107c5dd"
tuned_rf = tune_model(rf)

# + colab_type="code" id="nwaZk6oTEQsi" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="89fcfb21-9c75-472c-b99e-f30eb4062b23"
predict_model(tuned_rf);

# + [markdown] colab_type="text" id="r79BGjIfEQs1"
# # 12.0 Finalize Model for Deployment

# + colab_type="code" id="_--tO4KGEQs-" colab={}
final_rf = finalize_model(tuned_rf)

# + colab_type="code" id="U9W6kXsSEQtQ" colab={"base_uri": "https://localhost:8080/", "height": 161} outputId="304b6389-053e-42ea-a532-aff799a900a9"
#Final Random Forest model parameters for deployment
print(final_rf)

# + colab_type="code" id="NJDk3I-EEQtg" colab={"base_uri": "https://localhost:8080/", "height": 80} outputId="376e1d1e-3ab8-4c7e-ec47-9d115dcdb239"
predict_model(final_rf);

# + [markdown] id="dWU2Dmdx2UNZ" colab_type="text"
# # 13.0 Deploy Model on Microsoft Azure
#
# This is the code to deploy model on Microsft azure using `pycaret` functionalities.

# + id="PtdFIPJJ0zHX" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 485} outputId="565fc39a-5f4f-438b-9b39-a8fbff626aa6"
# # ! pip install azure-storage-blob
# ! pip install awscli



# + id="ImFnwpb52iDl" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="f2ebf115-0c20-4c8e-f082-4be8125822a4"
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
authentication = {'container': 'pycaret-cls-10111'}
model_name = 'rf-clf-101'
deploy_model(final_rf, model_name, authentication, platform = 'azure')

# + id="iuBz98UT2icD" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 107} outputId="c7da0fc9-71aa-40c5-9f73-24c5cf88a538"
authentication = {'container': 'pycaret-cls-10111'}
model_name = 'rf-clf-101'
model_azure = load_model(model_name, 
               platform = 'azure', 
               authentication = authentication,
               verbose=True)

# + id="aiP_EiLm2iWk" colab_type="code" colab={}

unseen_predictions = predict_model(model_azure, data=data_unseen, verbose=True)

# + id="UkX3mtAD2iJH" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 439} outputId="77e24380-a438-49a7-e7c8-ad3bde9e9ce1"
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

# + id="i8xWrcliQCz1" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 35} outputId="07407b18-4de9-47d8-83c2-1bad74e255a3"
# GCP project name, Change the name based on your own GCP project.
CLOUD_PROJECT = 'gcpessentials-rz' # GCP project name
bucket_name = 'pycaret-clf1011-test1' # bucket name for storage of your model
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

# + id="bIMlREBHXTtF" colab_type="code" colab={}

unseen_predictions = predict_model(model_gcp, data=data_unseen, verbose=True)

# + id="CFxn0KJ_ebGz" colab_type="code" colab={"base_uri": "https://localhost:8080/", "height": 439} outputId="00f63478-1f56-4180-dd51-f91b86dafb47"
unseen_predictions
