"""A standard machine learning task without much sacred magic."""
import glob
import logging
import os

from sacred import Experiment
from sacred import SETTINGS
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.observers.file_storage import DEFAULT_FILE_STORAGE_PRIORITY
from sacred.observers.mongo import DEFAULT_MONGO_PRIORITY

from pycaret.nlp import (
    setup,
    assign_model,
    create_model,
    models,
    plot_model,
    save_model
)

SETTINGS['CAPTURE_MODE'] = 'sys'  # work around for joblib timeout

fso = FileStorageObserver(
    basedir="my_runs",
    resource_dir=None,
    source_dir=None,
    template=None,
    priority=DEFAULT_FILE_STORAGE_PRIORITY
)
mongo_user = "mongo_user"
mongo_pwd = "mongo_password"
mongo_host = "localhost"
mongo_port = 27017
mongo_db = "sacred"
mongo_url = f"mongodb://{mongo_user}:{mongo_pwd}@{mongo_host}:{mongo_port}/{mongo_db}?authSource=admin"
mdbo = MongoObserver(
    url=mongo_url,
    db_name="sacred",
    collection="runs",
    overwrite=None,
    priority=DEFAULT_MONGO_PRIORITY,
    client=None,
    failure_dir="my_failures"
)

ex = Experiment("pycaret nlp on kiva dataset")
ex.observers.append(fso)
ex.observers.append(mdbo)
ex.add_config(
    dict(
        custom_stopwords=None,
        html=True,
        session_id=None,
        log_experiment=False,
        experiment_name=None,
        log_plots=False,
        log_data=False,
        verbose=True,
    )
)


@ex.main
def run(_config):
    logging.info("1. Loading Dataset")
    from pycaret.datasets import get_data
    data = get_data('kiva')

    logging.info("2. Initialize Setup")

    nlp1 = setup(
        data=data,
        target='en',
        custom_stopwords=_config['custom_stopwords'],
        html=_config['html'],
        session_id=_config['session_id'],
        log_experiment=_config['log_experiment'],
        experiment_name=_config['experiment_name'],
        log_plots=_config['log_plots'],
        log_data=_config['log_data'],
        verbose=_config['verbose'],
    )

    logging.info("3. Create Model")
    logging.debug("%r", f"available models = {models()}")
    logging.info("Latent Dirichlet Allocation")

    lda = create_model('lda')

    logging.info("Non-Negative Matrix Factorization")
    nmf = create_model('nmf', num_topics=4)

    logging.info("Latent Semantic Indexing")
    lsi = create_model("lsi")

    logging.info("Hierarchical Dirichlet Process")
    hdp = create_model("hdp")

    logging.info("Random Projections")
    rp = create_model("rp")

    logging.info("4. Assign Labels")
    lda_results = assign_model(lda)
    nmf_results = assign_model(nmf)
    lsi_results = assign_model(lsi)
    hdp_results = assign_model(hdp)
    rp_results = assign_model(rp)

    logging.info("5. Analyze Model")
    logging.debug("""
    available plots:
    * Word Token Frequency - 'frequency'
    * Word Distribution Plot - 'distribution'
    * Bigram Frequency Plot - 'bigram' 
    * Trigram Frequency Plot - 'trigram'
    * Sentiment Polarity Plot - 'sentiment'
    * Part of Speech Frequency - 'pos'
    * t-SNE (3d) Dimension Plot - 'tsne'
    * Topic Model (pyLDAvis) - 'topic_model' # only works in notebook
    * Topic Infer Distribution - 'topic_distribution'
    * Wordcloud - 'wordcloud'
    * UMAP Dimensionality Plot - 'umap'
    """)

    for plot_type in [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_distribution",
        "wordcloud",

    ]:
        plot_model(lda, plot=plot_type, save=True)
        for image in glob.glob("*.html"):
            ex.add_artifact(filename=image, name=f"lda_{image}")
            os.remove(image)
        for image in glob.glob("*.png"):
            ex.add_artifact(filename=image, name=f"lda_{image}")
            os.remove(image)

    for plot_type in [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_distribution",
        "wordcloud",
    ]:
        plot_model(nmf, plot=plot_type, save=True)
        for image in glob.glob("*.html"):
            ex.add_artifact(filename=image, name=f"nmf_{image}")
            os.remove(image)
        for image in glob.glob("*.png"):
            ex.add_artifact(filename=image, name=f"nmf_{image}")
            os.remove(image)

    for plot_type in [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_distribution",
        "wordcloud",
    ]:
        plot_model(lsi, plot=plot_type, save=True)
        for image in glob.glob("*.html"):
            ex.add_artifact(filename=image, name=f"lsi_{image}")
            os.remove(image)
        for image in glob.glob("*.png"):
            ex.add_artifact(filename=image, name=f"lsi_{image}")
            os.remove(image)

    for plot_type in [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_distribution",
        "wordcloud",

    ]:
        plot_model(hdp, plot=plot_type, save=True)
        for image in glob.glob("*.html"):
            ex.add_artifact(filename=image, name=f"hdp_{image}")
            os.remove(image)
        for image in glob.glob("*.png"):
            ex.add_artifact(filename=image, name=f"hdp_{image}")
            os.remove(image)

    for plot_type in [
        "frequency",
        "distribution",
        "bigram",
        "trigram",
        "sentiment",
        "pos",
        "tsne",
        "topic_distribution",
        "wordcloud",
    ]:
        plot_model(rp, plot=plot_type, save=True)
        for image in glob.glob("*.html"):
            ex.add_artifact(filename=image, name=f"rp_{image}")
            os.remove(image)
        for image in glob.glob("*.png"):
            ex.add_artifact(filename=image, name=f"rp_{image}")
            os.remove(image)

    logging.info("7. Save / Load Model")

    save_model(lda, model_name='lda-model')
    save_model(nmf, model_name='nmf-model')
    save_model(lsi, model_name='lsi-model')
    save_model(hdp, model_name='hdp-model')
    save_model(rp, model_name='rp-model')
    for pkl in glob.glob("*.pkl"):
        ex.add_artifact(pkl)
        os.remove(pkl)


def cleanup():
    for pkl in glob.glob("*.pkl"):
        os.remove(pkl)
    for png in glob.glob("*.png"):
        os.remove(png)
    for html in glob.glob("*.html"):
        os.remove(html)


if __name__ == "__main__":
    cleanup()
    ex.run()
