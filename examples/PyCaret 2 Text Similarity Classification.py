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

# # Text Similarity Classification
# <i>Last Update: 08/02/2020</i><br>
# <i>PyCaret Version: 2.0</i><br>
# <i>Author: Alexandre Farias</i><br>
# <i>Email: afarias@tuta.io</i>
#
# # Introduction
# This task consists in compare two sentences present on the dataset and identify if both have the same meaning.<br>
# An Exploratory Data Analysis is made to gain insights about the data, a Topic Modelling to get the features and the classification step, these last two steps are made with the Python Module PyCaret.<br>
# Importing the requires modules, PyCaret is imported later to avoid conflicts on the experiments.
# The dataset used on this work is the [Text Similarity](https://www.kaggle.com/rishisankineni/text-similarity) hosted on Kaggle, only using the data present on the train set, since the test set has a few samples doesn't has the labels do validate the model.

# Standard
import pandas as pd
# Plots
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
init_notebook_mode(connected=True)
import seaborn as sns
# Sklearn tools
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
# PATH and setup
import os
os.chdir("..")
PATH = os.getcwd()+os.sep
RANDOM_SEED = 42
K_FOLDS = 5


# The helper functions used to plots, data sampling and scores.

## Dataset Sampling
def data_sampling(dataset, frac: float, random_seed: int):
    data_sampled_a = dataset.sample(frac=frac,
                                    random_state=random_seed)
    data_sampled_b =  dataset.drop(data_sampled_a.index).\
    reset_index(drop=True)
    data_sampled_a.reset_index(drop=True, inplace=True)
    return data_sampled_a, data_sampled_b  
## Pie Chart
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
                      marker=dict(line=dict(color="#000000",
                      width=2)))
    fig.update_yaxes(automargin=True)
    iplot(fig)
## Histogram subplots
def histogram_subplot(dataset_a, dataset_b, feature_a: str,
feature_b: str, title: str, title_a:
str, title_b: str):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(
    title_a,
    title_b
    )
    )
    fig.add_trace(go.Histogram(x=dataset_a[feature_a],
    showlegend=False), row=1, col=1)
    fig.add_trace(go.Histogram(x=dataset_b[feature_b],
    showlegend=False), row=1, col=2)
    fig.update_layout(template="simple_white")
    fig.update_layout(autosize=False,
    title={"text" : title,
            "y" : 0.9,
            "x" : 0.5,
            "xanchor" : "center",
            "yanchor" : "top"},
            yaxis={"title" : "<i>Frequency</i>"})
    fig.update_traces(marker=dict(line=dict(color="#000000",
    width=2)))
    fig.update_yaxes(automargin=True)
    iplot(fig)
# Calculate scores with Test/Unseen labeled data
def test_score_report(data_unseen, predict_unseen):
    le = LabelEncoder()
    data_unseen["Label"] = le.fit_transform(data_unseen.same_security.values)
    data_unseen["Label"] = data_unseen["Label"].astype(int)
    accuracy = accuracy_score(data_unseen["Label"],
    predict_unseen["Label"])
    roc_auc = roc_auc_score(data_unseen["Label"],
    predict_unseen["Label"])
    precision = precision_score(data_unseen["Label"],
    predict_unseen["Label"])
    recall = recall_score(data_unseen["Label"],
    predict_unseen["Label"])
    f1 = f1_score(data_unseen["Label"],
    predict_unseen["Label"])
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
                    predict_unseen["Label"], labels=unique_label),
                    index=['{:}'.format(x) for x in unique_label],
                    columns=['{:}'.format(x) for x in unique_label])
    ax = sns.heatmap(cmtx, annot=True, fmt="d", cmap="YlGnBu")
    ax.set_ylabel('Predicted')
    ax.set_xlabel('Target');
    ax.set_title("Predict Unseen Confusion Matrix", size=14);


# # 1. Data Loading and Initial Infos
# The data is loaded in a Dataframe.

data = pd.read_csv(PATH+"data"+os.sep+"train.csv")
data.head(3)

# A check for duplicated samples.

data.drop_duplicates(inplace=True, keep='first')

# There're no duplicated samples on the dataset.<br>
# Time to drop features that will not help on the prediction and get some information about the data.

data.drop(columns=['Unnamed: 0', 'ticker_x', 'ticker_y'],
          inplace=True)
data.info()

# And check for missing values.

data.isnull().sum()

# The dataset doesn't contain any missing value, what is very good!
# A good approach to compare the sentences is check its characters length, let's create these features for each sentence.

data['len_x'] = data['description_x'].apply(len)
data['len_y'] = data['description_y'].apply(len)

# # 2. EDA

# Let's check the distribution of the target feature, `same_security`.

series_aux = data['same_security'].value_counts()
pie_plot(series_aux, 'Security Values')

# It's a case of imbalanced classes and it will be handled with a model evaluation using F1-Score. I will not use any resampling technique, as random oversampling, SMOTE, etc. <br>
# A good insight for this problem is see the distribution of character length of each sentence for the security values, starting with the True values.

data_true = data.query('same_security == True')
histogram_subplot(data_true, data_true, 'len_x', 'len_y', 
                  '<b>Character length distribution for True Security', 
                  'Description X Length', 'Description Y Length')

# Sentences that are True on the comparing, have a similar distribution, even on the frequency for the values, for now, seems a good idea use the character length as a feature.<br>
# To make this idea better, let's check it for the False values.

data_false = data.query('same_security == False')
histogram_subplot(data_false, data_false, 'len_x', 'len_y', 
                  '<b>Character length distribution for False Security', 
                  'Description X Length', 'Description Y Length')

# And for the False values the distribution for each sentence is different, I will use both features about the character length to feed the model.

# # 3. Model Build
# PyCaret has a NLP module that can automatize most of the boring stuff, like lowering case, remove stop words, stemming, etc. So, a good amount of this part is just setting up PyCaret to run.<br>
# Let's import the module.

from pycaret.nlp import *

# The PyCaret setup is simple, just enter data, the text target to process and a random seed for the session.<br>
# It's possible to add custom stop words, but I will use the default from PyCaret.<br>
# Let's start with the text from `description_x`.

exp_x = setup(data=data, target='description_x', session_id=RANDOM_SEED)

# There are 310 words present in the vocabulary for `sentence_x`.<br>
# To get the features for the model, is used a Topic Modelling with PyCaret, for it, the model is tuned (LDA Model) to see what is the best number of topics.

tuned_model_x = tune_model(model='lda', supervised_target='same_security')

# Based on the F1-Score, the model doesn't have a high variation on the score, so I will use 4 topics, to keep it simples.<br>
# Time to create a LDA model with 4 topics and see some informations about the topic modelled data.<br>
# You can click on each box to see infos like frequency of a word, word cloud, etc.<br>

model_x = create_model('lda', num_topics=4)

# The model is assigned to the topic weights and other infos to use after in the classification task.

model_x_results = assign_model(model_x)

# Now, all the steps are made again for the `description_y`.

exp_y = setup(data=data, target='description_y', session_id=RANDOM_SEED)

# Again, 310 words are present in the vocabulary for `sentence_y`.

tuned_model_y = tune_model(model='lda', supervised_target='same_security')

# The result from the tuned model is similar from before, so 4 topics is used again.

model_y = create_model('lda', num_topics=4)

# And assign the topics to the model.

model_y_results = assign_model(model_y)

# Topic Modelling made and is time to make the classification.

# # 4. Classification
# To start, the unused columns are dropped and the results are concatenated in one dataframe.

data_topics = model_x_results.drop(columns=['description_x', 'description_y', 
                                         'Dominant_Topic', 'Perc_Dominant_Topic'])
data_topics['Topic_0_y'] = model_y_results['Topic_0']
data_topics['Topic_1_y'] = model_y_results['Topic_1']
data_topics['Topic_2_y'] = model_y_results['Topic_2']
data_topics['Topic_3_y'] = model_y_results['Topic_3']

# To avoid errors with PyCaret on the label feature, `same_security` is converted to object and the labels are renamed to `Yes` for `True` and `No` for `False`.<br>
# 10% of the data is sampled to be used as unseen data to validate the final model.

# +
from pycaret.classification import *

data_topics['same_security'] = data_topics['same_security'].astype('str')
data_topics_dict = {'True' : 'Yes',  'False' : 'No'}
data_topics['same_security'] = data_topics['same_security'].replace(data_topics_dict)
train, unseen = data_sampling(data_topics, 0.9, RANDOM_SEED)
data_topics.head(5)
# -

# The same setting up from before, but now PyCaret split the dataset in 70% for Train and 30% for Test.<br>
# No feature transformation, scalling or normalization are used.

exp_clf = setup(data=train, 
                target='same_security',
                session_id=RANDOM_SEED, 
                experiment_name="text-similarity",
                log_experiment=True,
                silent=True)

# Compare the models sorted by F1-Score to get the best.`

top_model = compare_models(sort='F1',
                           fold=K_FOLDS,
                           n_select=3)

# The best model by F1-Score is the Catboost Classifier, but the difference from XGBoost is minimal and this model got a better score on the other metrics, let's take XGBoost as the base model.
# Tune the model to see if can get any improvement.

tuned_model = tune_model(top_model[1], optimize='F1',
                         choose_better=True, fold=K_FOLDS);

# The tuned model doesn't get any improvement, so the base model is the best.<br>
# Time to build a Bagging Ensemble.

bagged_model = ensemble_model(tuned_model, optimize="F1",
                              fold=K_FOLDS) 

# And now a Boosting Ensemble.

boosted_model = ensemble_model(tuned_model, optimize="F1",
                               fold=K_FOLDS, method="Boosting") 

# The Bagged Model is the best and is saved as the best model and used to predict on the test set.

best_model = bagged_model
predict_model(best_model);

# Let's check the model hyperparameters.

plot_model(best_model, plot="parameter")

# Now, plot the AUC Score.

plot_model(best_model, plot="auc")

# The AUC Score for both classes was good, 0.87.<br>
# Now, the confusion matrix and class report.

plot_model(best_model, plot="confusion_matrix")

plot_model(best_model, plot="class_report")

# And the results are excellent for the class True (1), but good for the class False (0). <br>
# As the test data is well fitted on the model, let's use it to fit a final model.

final_model = finalize_model(best_model)

# # 5. Validation on Unseen Data
# To validadte the model, let's see the prediction with unseen data, which was not included on the final model fit.

predict_unseen = predict_model(final_model, data=unseen);
score_unseen = test_score_report(unseen, predict_unseen)
print(score_unseen.to_string(index=False))
conf_mat(unseen, predict_unseen);

# So, the unseen data was well predicted on the model, with a high F1-Score showing a good balance between Recall and precision.

# # 6. Conclusion
# The following conclusions can be drawed:
# * The Model just needed 4 Topics Modelled to compare the sentences.
# * Character length is important to compare the sentences.
# * Recall was excellent, getting about 95% of the positive labels.
# * Precision was excellent too, predicting correctly about 90% of the values labeled as positives.
# * F1-Score was about 92% with a good balance between recall and precision.
# * The Length on the `sentence_y` is important to model prediction performance.
