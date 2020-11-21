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

# + [markdown] cell_id="7088943d-f7b9-4b97-b640-6a775e1e8450" tags=[]
# # Customer Churn Prediction
# <i>Last Update: 08/02/2020</i><br>
# <i>PyCaret Version: 2.0</i><br>
# <i>Author: Alexandre Farias</i><br>
# <i>Email: afarias@tuta.io</i>
#
# ![](https://res.cloudinary.com/dn1j6dpd7/image/fetch/f_auto,q_auto,w_736/https://www.livechat.com/wp-content/uploads/2016/04/customer-churn@2x.jpg)
# <i>Image Source:</i> [What Is Churn Rate and Why It Will Mess With Your Growth](https://www.livechat.com/success/churn-rate/)

# + [markdown] cell_id="9b238085-0cbf-4c9e-afa5-88b72a835fae" tags=[]
# ## 1. Introduction
# Customer Churn is when customers leave a service in a given period of time, which is bad for business.<br>
# This work has as objective to build a machine learning model to predict which customers will leave the service, the dataset used on this notebook is the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) hosted at Kaggle. Also, an Exploratory Data Analysis is made to a better understand about the data. 
# Another point on this work is use Deepnote as development enviroment and the [PyCaret 2.0](https://pycaret.org/) Python Module to make all the experiment pipeline. 

# + [markdown] cell_id="1e78a237-948f-480d-ba96-3503917949ad" tags=[]
# ### 1.1 Enviroment Setup
# The Modules used for this work, highlights for PyCaret 2.0 and good plots by [Plotly](https://plotly.com/).

# + cell_id="464a9356-824f-4c4d-9945-224698b09877" output_cleared=false
# Standard
import pandas as pd
import numpy as np
import os
# Pycaret
from pycaret.classification import *
# Plots
from plotly.offline import iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
# Sklearn tools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
# Extras
from datetime import date
import warnings
warnings.filterwarnings("ignore")
# Datapath and Setup
os.chdir("..")
PATH = os.getcwd()+os.sep
RANDOM_SEED = 142
K_FOLDS = 5


# + [markdown] cell_id="2bcfc94d-a561-405d-ac35-65ac706a54dc" tags=[]
# And the helper functions used on this notebook.

# + cell_id="28308802-8246-4878-aa00-e65ee5bd184a" tags=[]
# Helper functions for structured data
## Get info about the dataset
def dataset_info(dataset, dataset_name: str):
    print(f"Dataset Name: {dataset_name} \
        | Number of Samples: {dataset.shape[0]} \
        | Number of Columns: {dataset.shape[1]}")
    print(30*"=")
    print("Column             Data Type")
    print(dataset.dtypes)
    print(30*"=")
    missing_data = dataset.isnull().sum()
    if sum(missing_data) > 0:
        print(missing_data[missing_data.values > 0])
    else:
        print("No Missing Data on this Dataset!")
    print(30*"=")
    print("Memory Usage: {} MB".\
         format(np.round(
         dataset.memory_usage(index=True).sum() / 10e5, 3
         )))
## Dataset Sampling
def data_sampling(dataset, frac: float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac, random_state=random_seed)
    data_sampled_b =  dataset.drop(data_sampled_a.index).reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b   
## Bar Plot
def bar_plot(data, plot_title: str, x_axis: str, y_axis: str):
    colors = ["#0080ff",] * len(data)
    colors[0] = "#ff8000"
    trace = go.Bar(y=data.values, x=data.index, text=data.values, 
                    marker_color=colors)
    layout = go.Layout(autosize=False, height=600,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"},  
                    xaxis={"title" : x_axis},
                    yaxis={"title" : y_axis},)
    fig = go.Figure(data=trace, layout=layout)
    fig.update_layout(template="simple_white")
    fig.update_traces(textposition="outside",
                    textfont_size=14,
                    marker=dict(line=dict(color="#000000", width=2)))                
    fig.update_yaxes(automargin=True)
    iplot(fig)
## Plot Pie Chart
def pie_plot(data, plot_title: str):
    trace = go.Pie(labels=data.index, values=data.values)
    layout = go.Layout(autosize=False,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"})
    fig = go.Figure(data=trace, layout=layout)
    fig.update_traces(textfont_size=14,
                    marker=dict(line=dict(color="#000000", width=2)))
    fig.update_yaxes(automargin=True)            
    iplot(fig)
## Histogram
def histogram_plot(data, plot_title: str, y_axis: str):
    trace = go.Histogram(x=data)
    layout = go.Layout(autosize=False,
                    title={"text" : plot_title,
                       "y" : 0.9,
                       "x" : 0.5,
                       "xanchor" : "center",
                       "yanchor" : "top"},  
                    yaxis={"title" : y_axis})
    fig = go.Figure(data=trace, layout=layout)
    fig.update_traces(marker=dict(line=dict(color="#000000", width=2)))
    fig.update_layout(template="simple_white")
    fig.update_yaxes(automargin=True)
    iplot(fig)
# Particular case: Histogram subplot (1, 2)
def histogram_subplot(dataset_a, dataset_b, feature_a: str,
                        feature_b: str, title: str, title_a: str, title_b: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
                        title_a,
                        title_b
                        )
                    )
    fig.add_trace(go.Histogram(x=dataset_a[feature_a],
                               showlegend=False),
                                row=1, col=1)
    fig.add_trace(go.Histogram(x=dataset_b[feature_b],
                               showlegend=False),
                              row=1, col=2)
    fig.update_layout(template="simple_white")
    fig.update_layout(autosize=False,
                        title={"text" : title,
                        "y" : 0.9,
                        "x" : 0.5,
                        "xanchor" : "center",
                        "yanchor" : "top"},  
                        yaxis={"title" : "<i>Frequency</i>"})
    fig.update_traces(marker=dict(line=dict(color="#000000", width=2)))
    fig.update_yaxes(automargin=True)
    iplot(fig)
# Calculate scores with Test/Unseen labeled data
def test_score_report(data_unseen, predict_unseen):
    le = LabelEncoder()
    data_unseen["Label"] = le.fit_transform(data_unseen.Churn.values)
    data_unseen["Label"] = data_unseen["Label"].astype(int)
    accuracy = accuracy_score(data_unseen["Label"], predict_unseen["Label"])
    roc_auc = roc_auc_score(data_unseen["Label"], predict_unseen["Label"])
    precision = precision_score(data_unseen["Label"], predict_unseen["Label"])
    recall = recall_score(data_unseen["Label"], predict_unseen["Label"])
    f1 = f1_score(data_unseen["Label"], predict_unseen["Label"])

    df_unseen = pd.DataFrame({
        "Accuracy" : [accuracy],
        "AUC" : [roc_auc],
        "Recall" : [recall],
        "Precision" : [precision],
        "F1 Score" : [f1]
    })
    return df_unseen
# Confusion Matrix
def conf_mat(data_unseen, predict_unseen):
    unique_label = data_unseen["Label"].unique()
    cmtx = pd.DataFrame(
        confusion_matrix(data_unseen["Label"],
                         predict_unseen["Label"],
                         labels=unique_label), 
        index=['{:}'.format(x) for x in unique_label], 
        columns=['{:}'.format(x) for x in unique_label]
    )
    ax = sns.heatmap(cmtx, annot=True, fmt="d", cmap="YlGnBu")
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Target');
    ax.set_title("Predict Unseen Confusion Matrix", size=14);


# + [markdown] cell_id="5ae1c017-271e-410e-abd5-0be3b8727766" tags=[]
# ## 2. Load Data
#
# The Dataset is load as a Pandas dataframe and show a gimplse of the data.
# A good thing about Deepnote is that the displayed dataframes shows the column type, helping to understand the features.

# + cell_id="cec0aaf1-7373-484e-bcdd-e39fa067e8ad" output_cleared=false tags=[]
dataset = pd.read_csv(PATH+"data"+os.sep+"customers.csv")
dataset.head(3)

# + [markdown] cell_id="6ec2b507-2382-46dc-a15e-b0795b6c6e6e" tags=[]
# Check for duplicated samples.

# + cell_id="8827849f-c74e-419a-adff-0a0d71e75f1c" tags=[]
dataset[dataset.duplicated()]

# + [markdown] cell_id="9b3cb07f-e86e-494a-87ec-04aeafbc5144" tags=[]
# There are no duplicated samples on the dataset.<br>
# More information about the dataset is needed as the number of samples, memory size allocation, etc.<br>
# The result is showed on the following output (The Data Type is just showed for convenience, to make this notebook useful on other enviroments).

# + cell_id="7a4a0d96-1085-4179-9523-fe1774edff40" tags=[]
dataset_info(dataset, "customers")

# + [markdown] cell_id="f8a7621d-34b8-49d8-85a6-ac7b71a545ba" tags=[]
# The dataset has a small memory size allocation (1.183 MB) and is composed for many Categorical (object) features and only a few numeric, but one of the categorical features doesn't look right, the `TotalCharges`, as showed on the displayed dataframe, the festure is numeric.<br>
# `TotalCharges` is converted from Object to float64, the same of `MonthlyCharges` feature.

# + cell_id="3520f897-c87b-4a90-9814-8edfd3220119" tags=[]
dataset["TotalCharges"] = pd.to_numeric(dataset["TotalCharges"], errors="coerce")
print(f"The Feature TotalCharges is type {dataset.TotalCharges.dtype} now!")

# + [markdown] cell_id="74bc6ebf-5467-4c13-af9f-4d2744ba11ec" tags=[]
# ## 3. Exploratory Data Analysis

# + [markdown] cell_id="aceeb11e-7a7d-4367-9492-1714689c6c5a" tags=[]
# ### 3.1 Churn Distribution
# The Client Churn Distribution is checked for any imbalance, as the feature is the target, it's important to choose what strategy to adopt when dealing with imbalanced classes.<br>
# Below, a Pie Chart shows the feature distribution.
#

# + cell_id="0e06fe3a-30b9-4ff2-b803-951dd0b8e46c" tags=[]
pie_plot(dataset["Churn"].value_counts(), 
         plot_title="<b>Client Churn Distribution<b>")

# + [markdown] cell_id="e30ae2f1-f982-462f-80f3-1bc64e1d31d1" tags=[]
# There's some imbalance on Churn Distribution, 26.5% of the clients have churned, and small occurences of a label could lead to bad predictor.<br>
# It's possible to choose some ways to work with this case:
# * Make a random over-sampling, duplicating some samples of the minority class until this reach a balance, but this could lead to an overfitted model.
# * Make a random down-sampling, removing some samples from the majority class until this reach a balance, but this leads to information loss and not feeding the model with the collected samples.
# * Make a random down-sampling, removing some samples from the majority class until this reach a balance, but this leads to information loss and not feeding the model with the collected samples.
# * Another resampling technique, as SMOTE.
# * Choosing a metric that deals with imbalanced datasets, like F1 Score.
#
# The Churn problem is about client retention, is worth to check about false positives, so precision and recall metrics are a must for this situtation.<br>
# F1 Score is used to check the quality of the model predictions, as the metric is an harmonic mean of precision and recall.

# + [markdown] cell_id="61904ad8-8f19-4ed5-bf91-8fbabcb40de1" tags=[]
# ### 3.2 Analysis of the Contract Type
#

# + [markdown] cell_id="84546349-680a-41c9-9a0f-3dc2f376a8b2" tags=[]
# The contract type is a good feature to analyze what happens to a client churn from that service, a plot from the contract types of not churned clients is showed below.

# + cell_id="b560ac2d-e6c1-49ec-980e-da39712c2f85" tags=[]
df_aux = dataset.query('Churn == "No"')
df_aux = df_aux["Contract"].value_counts()
bar_plot(df_aux, "<b>Contract Types of not Churned Clients</b>",
         "<i>Contract</i>", "<i>Count</i>")

# + [markdown] cell_id="a5778511-9798-45f7-8022-db6210e76fc6" tags=[]
# Is showed that a Month-to-month contract is the firts when compared to annual contracts, but the difference between the number of contracts is not so big.<br>
# To a better comparation, the same plot is showed for the churned clients.

# + cell_id="d73d0256-a7f9-48af-9b08-741ece81ff43" tags=[]
df_aux = dataset.query('Churn == "Yes"')
df_aux = df_aux["Contract"].value_counts()
bar_plot(df_aux, "<b>Contract Types of Churned Clients</b>",
         "<i>Contract</i>", "<i>Count</i>")

# + [markdown] cell_id="da76f779-776c-4d75-bd18-d0e9c4e8e439" tags=[]
# Now, the difference between a Month-to-month and annual contractts is bigger, and can lead to a conclusion that annual contracts are better to retain the clients, perhaps fidelity promotions could aid to reduce the churn rate.<br>
# As the problem can be examined more deep on Month-to-month contract types, a good idea is see the Monthly Charges and Total Charges distribution for the not churned clients of this contract.
#

# + cell_id="632a627e-7764-4629-9408-93e86f15d99e" tags=[]
df_aux = dataset.query('(Contract == "Month-to-month") and (Churn == "No")')
histogram_subplot(df_aux, df_aux, "MonthlyCharges", "TotalCharges", 
                  "<b>Charges Distribution for Month-to-month contracts for not Churned Clients</b>",
                  "(a) Monthtly Charges Distribution", "(b) Total Charges Distribution")

# + [markdown] cell_id="af9de48f-100b-441c-98c0-76a28a29fc46" tags=[]
# From the plots, can be said that many clients just got charged with a few values, principally for the Total Charges.<br>
# On the following plots, the same features are analyzed, but for churned clients.

# + cell_id="3a7d59f0-0bfa-40a1-80bc-d8208382cf9c" tags=[]
df_aux = dataset.query('(Contract == "Month-to-month") and (Churn == "Yes")')
histogram_subplot(df_aux, df_aux, "MonthlyCharges", "TotalCharges", 
                  "<b>Charges Distribution for Month-to-month contracts for Churned Clients</b>",
                  "(a) Monthtly Charges Distribution", "(b) Total Charges Distribution")

# + [markdown] cell_id="0a9867c9-7fbc-491c-ac50-c033404f654f" tags=[]
# Total Charges had the same behaviour, but the Monthly Charges for many churned clients was high, maybe the amount of chage value could lead the client to leave the service.<br>
# Still on the Month-to-month contract, it's time to analyze the most used Payment methods of churned clients.

# + cell_id="5b9d7be6-7b0a-4276-bf80-0e955af80524" tags=[]
df_aux = dataset.query(('Contract == Month-to-month') and ('Churn == "Yes"'))
df_aux = df_aux["PaymentMethod"].value_counts()
bar_plot(df_aux, "<b>Payment Method of Month-to-month contract Churned Clients</b>",
         "<i>Payment Method</i>", "<i>Count</i>")

# + [markdown] cell_id="0b85ae6a-63cf-457b-a105-f013d1798415" tags=[]
# Many Churned Clients used to pay with electronic checks, automatic payments, as bank transfers or credit card have a few churned clients. A good idea could make promotions to clients that use automatic payment methods. <br>
# Lastly, the tenure of the churned clients.

# + cell_id="0bcdd312-51fd-4748-8035-889060017e5c" tags=[]
df_aux = dataset.query(('Contract == Month-to-month') and ('Churn == "Yes"'))
df_aux = df_aux["tenure"].value_counts().head(5)
bar_plot(df_aux, "<b>Tenure of Month-to-month contract for Churned Clients</b>",
         "<i>Tenure</i>", "<i>Count</i>")

# + [markdown] cell_id="007f6510-4a74-46a4-952b-225af9f37d03" tags=[]
# Most clients just used the service for one month, seems like the clients used to service to check the quality or the couldn't stay for the amount of charges, as the Monthly Charges for these clients was high and the Total Charges was small, as the client just stayed a little time.  

# + [markdown] cell_id="8546f71b-652c-4909-b31b-b11cd7cc52c1" tags=[]
# ## 4. Setting up PyCaret

# + [markdown] cell_id="f96ae182-aead-4514-af36-6ffcf5df7e2e" tags=[]
# Before setting up PyCaret, a random sample of 10% size of the dataset will be get to make predictions with unseen data. 

# + cell_id="d6011a29-86c1-44e5-b754-8d952ddce4cc" tags=[]
data, data_unseen = data_sampling(dataset, 0.9, RANDOM_SEED)
print(f"There are {data_unseen.shape[0]} samples for Unseen Data.")

# + [markdown] cell_id="506eea27-2ffc-48ba-a05f-19ba290d3d1e" tags=[]
# The PyCaret's setup is made with 90% of data samples and just use one function (`setup`) from the module.<br>
# It's possible configure with variuos options, as data pre-processing, feature engineering, etc. The easy and efficient of PyCaret buy a lot of time when prototyping models.<br>
# Each setup is an experiment and for this problem, is used the following options:
# * Normalization of the numerical features with Z-Score.
# * Feature Selection with permutation importance techniques.
# * Outliers Removal.
# * Features Removal based on Multicollinearity.
# * Features Scalling Transformation.
# * Ignore low variance on Features.
# * PCA for Dimensionality Reduction, as the dataset has many features.
# * Numeric binning on the features `MonthlyCharges` and `TotalCharges`.
# * 70% of samples for Train and 30% for test.
# * Fix Imbalance with SMOTE.
# * The models will store their metric results via Mlflow, to acess Mlflow UI, type mlflow on the your cmd or !mlflow ui on a cell on the bottom of this notebook. 

# + cell_id="6f4ed29e-1357-47b2-a4e4-637ce4119bbc" tags=[]
exp01 = setup(data=data, target="Churn", session_id=RANDOM_SEED, ignore_features=["customerID"], 
                numeric_features=["SeniorCitizen"], normalize=True,
                feature_selection=True, remove_outliers=True,
                remove_multicollinearity=True, fix_imbalance=True,
                transformation=True, ignore_low_variance=True, pca=True, 
                bin_numeric_features=["MonthlyCharges", "TotalCharges"],
                silent=True, experiment_name="customer-churn-prediction",
                log_experiment=True)

# + [markdown] cell_id="8d33c0f2-51f2-4b29-8548-7b354dd14b00" tags=[]
# PyCaret shows at first if all features types are with it correspondent type, if everything is right, press enter on the blank bar and the setup is finished showing a summary of the experiment. 

# + [markdown] cell_id="4ae241d1-57c6-4dcc-91c0-5d6f10b11bf1" tags=[]
# ## 5. Model Build

# + [markdown] cell_id="cc8f9047-81a4-46cc-902e-5450348b5b33" tags=[]
# A great tool on PyCaret is build many models and compare a metric for the bests! <br>
# The models are sorted by F1 Score due Precision and Recall are importants for the evaluation.<br>
# The cross-validation is made with 5-folds.

# + cell_id="922945dc-4001-4a47-b782-9515570044cf" tags=[]
top_model = compare_models(fold=K_FOLDS,
                            sort="F1",
                            n_select=1, 
                            blacklist=["gbc", "catboost"])

# + [markdown] cell_id="0bdc63ac-21e4-4d6f-afa3-a12cce4e2023" tags=[]
# The best model suggested by PyCaret is the Logistic Regreesion, with a F1 Score around 0.63 and a good Recall, around 0.77.<br>
# Time to tune the model. The `choose_better` argument get the best model between the tuned and best model.

# + cell_id="be5c3da4-91f5-43f1-a5dd-a36802be9dc5" tags=[]
tuned_model = tune_model(estimator=top_model, fold=K_FOLDS,
                         optimize="F1", choose_better=True,
                         verbose=False)
# -

# Let's see the hyperparameters of the chosen model

plot_model(tuned_model, plot="parameter")

# + [markdown] cell_id="590b1194-08b9-4c9e-9329-9b1efab65302" tags=[]
# PyCaret also has functions to make ensembles, for this implementation, a bagged model is build.

# + cell_id="66bf4246-2a40-494b-bd2e-69bd8897ff23" tags=[]
bagged_model = ensemble_model(tuned_model, fold=K_FOLDS)
# -

# And now a boosted model.

boosted_model = ensemble_model(tuned_model, fold=K_FOLDS,
                               method="Boosting")

# + [markdown] cell_id="282646c1-03c4-4ad2-a39c-cc99f0fafddb" tags=[]
# The boosted model improved a bit the F1 Score, it's also possible make blended and stacked models with PyCaret, both models are created using the the tuned and boosted models.

# + cell_id="bb52afc0-9e6b-4be8-8e41-c51e0f88317c" tags=[]
blended_model = blend_models(estimator_list=[tuned_model, boosted_model],
                            fold=K_FOLDS)

# + [markdown] cell_id="a76c2608-fb6d-4ee9-b706-88391ff104cc" tags=[]
# The best model still is the boosted model.<br>
# Let's plot some metric curves, matrices and see what is the model classifier, starting with the hyperparameters and the used model classifier.
# -

best_model = boosted_model
plot_model(best_model, plot="parameter")
print(f"Model: {type(best_model)}")

# Let's plot the ROC curve, PR Curve, Confusion Matrix and Metrics for each class.

# + cell_id="b112eea9-4394-414c-8aec-00a046bc7782" tags=[]
plot_model(best_model, plot="auc")
# -

# The AUC for each class was good: 0.85.

plot_model(best_model, plot="pr")

# The PR curve got an average precision around 0.7, which is good.

plot_model(best_model, plot="confusion_matrix")

# The Confusion Matrix shows that the churned clients have been classified as not churn in 50% of the predictions

plot_model(best_model, plot="class_report")

# The model has done a good work on the metrics for the class 0 (Not Churned) but got a Precision close to 0.5 for class 1 (Churned).

# + [markdown] cell_id="c1d1172b-74f5-445e-bfa7-d06b05834269" tags=[]
# ## 6. Prediction on Test Data

# + [markdown] cell_id="3c3a666c-a86e-4315-ba38-c659e60f9a18" tags=[]
# The test is made with the remaining 30% of data that PyCaret got on the setup, it's important to see that the model is not overfitting.

# + cell_id="49f5255a-960e-4245-a7bd-6b86c5018e91" tags=[]
predict_model(best_model);

# + [markdown] cell_id="80c031f4-190f-4991-a95c-1a00055a80a6" tags=[]
# As everything is right with the model, it's time to finalize it fitting all the data.

# + cell_id="d0909c9b-9391-4ad0-bd22-f9a5bbcd679f" tags=[]
final_model = finalize_model(best_model)

# + [markdown] cell_id="4d403f19-594d-40d7-92be-84094ccac0ba" tags=[]
# ## 7. Prediction on Unseen Data

# + [markdown] cell_id="949f3cc4-95ff-45c2-ba2c-e9fdaa356aa6" tags=[]
# The remaining 10% data is used to make predictions with unseen samples, what could include some outliers, it's how real world data works.<br>
# Just Kappa Score is not showed, as the focus is the F1 Score, as Precision and Recall are importants to get False Positives and False Negatives.<br>
# It's not necessary to make any transformation on the data, PyCaret do this.

# + cell_id="f2774b74-956e-4869-a3fd-3b041c16f22e" tags=[]
predict_unseen = predict_model(final_model, data=data_unseen);
score_unseen = test_score_report(data_unseen, predict_unseen)
print(score_unseen.to_string(index=False))
conf_mat(data_unseen, predict_unseen)

# + [markdown] cell_id="44ccff9f-85ca-415a-8057-aec964fdf547" tags=[]
# And the Unseen Data predicts as the trained model! The model was sucessful built! 

# + [markdown] cell_id="ee75d6f1-e02e-410a-b705-27503849659e" tags=[]
# ## 8. Save Experiment and Model

# + [markdown] cell_id="52e2e634-b24b-4a3a-b75f-ed22f217d916" tags=[]
# PyCaret allows to save all the pipeline experiment and the model to deploy.<br>
# It's recommended to save with date of the experiments.

# + cell_id="a774416a-ff50-44c6-b542-eb25a3d4daa6" tags=[]
save_model(final_model, PATH+"models"+os.sep+"modelCCP_"+date.today().strftime("%m-%d-%Y"))

# + [markdown] cell_id="b62c91e5-db7d-49f8-9747-7e9f0c71cf47" tags=[]
# ## 9. Conclusion

# + [markdown] cell_id="90128e4b-60f7-4c8d-8add-f76174294f61" tags=[]
# From the results and explanations presented here, some conclusion can be draw:
# * The type of contract has a strict relationship with churned clients, Month-to-month contracts with high amount of charges could lead a client to leave the service.
# * The Best Model is an AdaBoostClassifier with Logistic Regression as Base Estimator.
# * For the predictions made by the model and based on the precision and recall scores, as F1 Score try to show a balance between these two metrics, the precision was near 50%, what means that the model predict correctly 50% of classified clients as churned, on other hand, the recall was good, where around 83% of the actually churned clients was predict correctly.
# * The metrics must be used in favor of the business interests: Is needed a more correct prediction of the churned clients or get a more amount of these clients on the predictions?
#
# From the tools and enviroment used:
# * PyCaret is incredible, it speed up the model build a lot and the pre-processing is very useful, beyond of the functions to save and deploy the model.
