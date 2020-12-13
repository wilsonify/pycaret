"""A standard machine learning task without much sacred magic."""
import glob
import logging
import os

from matplotlib import pyplot as plt
import sklearn
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.observers.file_storage import DEFAULT_FILE_STORAGE_PRIORITY
from sacred.observers.mongo import DEFAULT_MONGO_PRIORITY
from pycaret.datasets import get_data
from pycaret.clustering import (
    setup,
    create_model,
    models,
    plot_model,
    evaluate_model,
    predict_model,
    save_model,
    load_model,
    tune_model,
    get_config,
    set_config,
    assign_model
)

from sacred import SETTINGS

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

ex = Experiment("pycaret clustering on public health dataset")
ex.observers.append(fso)
ex.observers.append(mdbo)
ex.add_config(
    dict(
        preprocess=True,  #: bool = True,
        imputation_type="simple",  #: str = "simple",
        iterative_imputation_iters=5,  #: int = 5,
        categorical_features=None,  #: Optional[List[str]] = None,
        categorical_imputation="mode",  #: str = "mode",
        categorical_iterative_imputer="lightgbm",  #: Union[str, Any] = "lightgbm",
        ordinal_features=None,  #: Optional[Dict[str, list]] = None,
        high_cardinality_features=None,  #: Optional[List[str]] = None,
        high_cardinality_method="frequency",  #: str = "frequency",
        numeric_features=None,  #: Optional[List[str]] = None,
        numeric_imputation="mean",  #: str = "mean",
        numeric_iterative_imputer="lightgbm",  #: Union[str, Any] = "lightgbm",
        date_features=None,  #: Optional[List[str]] = None,
        ignore_features=['Country Name'],  #: Optional[List[str]] = None,
        normalize=False,  #: bool = False,
        normalize_method="zscore",  #: str = "zscore",
        transformation=False,  #: bool = False,
        transformation_method="yeo-johnson",  #: str = "yeo-johnson",
        handle_unknown_categorical=True,  #: bool = True,
        unknown_categorical_method="least_frequent",  #: str = "least_frequent",
        pca=False,  #: bool = False,
        pca_method="linear",  #: str = "linear",
        pca_components=None,  #: Optional[float] = None,
        ignore_low_variance=False,  #: bool = False,
        combine_rare_levels=False,  #: bool = False,
        rare_level_threshold=0.10,  #: float = 0.10,
        bin_numeric_features=None,  #: Optional[List[str]] = None,
        remove_multicollinearity=False,  #: bool = False,
        multicollinearity_threshold=0.9,  #: float = 0.9,
        remove_perfect_collinearity=False,  #: bool = False,
        group_features=None,  #: Optional[List[str]] = None,
        group_names=None,  #: Optional[List[str]] = None,
        n_jobs=-1,  #: Optional[int] = -1,
        use_gpu=False,  #: bool = False,
        custom_pipeline=None,  #: Union[Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]] = None,
        html=True,  #: bool = True,
        session_id=None,  #: Optional[int] = None,
        log_experiment=False,  #: bool = False,
        experiment_name=None,  #: Optional[str] = None,
        log_plots=False,  #: Union[bool, list] = False,
        log_profile=False,  #: bool = False,
        log_data=False,  #: bool = False,
        silent=True,  #: bool = False,
        verbose=True,  #: bool = True,
        profile=False,  #: bool = False,
        profile_kwargs=None,  #: Dict[str, Any] = None,

    )
)


@ex.main
def run(_config):
    data = get_data('public_health')

    logging.info("2. Initialize Setup")

    clu1 = setup(
        data=data,
        preprocess=_config["preprocess"],
        imputation_type=_config["imputation_type"],
        iterative_imputation_iters=_config["iterative_imputation_iters"],
        categorical_features=_config["categorical_features"],
        categorical_imputation=_config["categorical_imputation"],
        categorical_iterative_imputer=_config["categorical_iterative_imputer"],
        ordinal_features=_config["ordinal_features"],
        high_cardinality_features=_config["high_cardinality_features"],
        high_cardinality_method=_config["high_cardinality_method"],
        numeric_features=_config["numeric_features"],
        numeric_imputation=_config["numeric_imputation"],
        numeric_iterative_imputer=_config["numeric_iterative_imputer"],
        date_features=_config["date_features"],
        ignore_features=_config["ignore_features"],
        normalize=_config["normalize"],
        normalize_method=_config["normalize_method"],
        transformation=_config["transformation"],
        transformation_method=_config["transformation_method"],
        handle_unknown_categorical=_config["handle_unknown_categorical"],
        unknown_categorical_method=_config["unknown_categorical_method"],
        pca=_config["pca"],
        pca_method=_config["pca_method"],
        pca_components=_config["pca_components"],
        ignore_low_variance=_config["ignore_low_variance"],
        combine_rare_levels=_config["combine_rare_levels"],
        rare_level_threshold=_config["rare_level_threshold"],
        bin_numeric_features=_config["bin_numeric_features"],
        remove_multicollinearity=_config["remove_multicollinearity"],
        multicollinearity_threshold=_config["multicollinearity_threshold"],
        remove_perfect_collinearity=_config["remove_perfect_collinearity"],
        group_features=_config["group_features"],
        group_names=_config["group_names"],
        n_jobs=_config["n_jobs"],
        use_gpu=_config["use_gpu"],
        custom_pipeline=_config["custom_pipeline"],
        html=_config["html"],
        session_id=_config["session_id"],
        log_experiment=_config["log_experiment"],
        experiment_name=_config["experiment_name"],
        log_plots=_config["log_plots"],
        log_profile=_config["log_profile"],
        log_data=_config["log_data"],
        silent=_config["silent"],
        verbose=_config["verbose"],
        profile=_config["profile"],
        profile_kwargs=_config["profile_kwargs"],

    )

    logging.info("3. Create Model")

    models()

    kmeans = create_model('kmeans', num_clusters=4)

    kmodes = create_model('kmodes', num_clusters=4)

    logging.info("4. Assign Labels")

    kmeans_results = assign_model(kmeans)
    kmeans_results.head()

    logging.info("5. Analyze Model")

    plot_model(kmeans, plot="cluster", save=True)

    plot_model(kmeans, feature='Country Name', label=True, save=True)

    plot_model(kmeans, plot='tsne', save=True)

    plot_model(kmeans, plot='elbow', save=True)

    try:
        plot_model(kmeans, plot='silhouette', save=True)
    except TypeError:
        logging.warning("unable to create silhouette plot")

    try:
        plot_model(kmeans, plot='distance', save=True)
    except TypeError:
        logging.warning("unable to create distance plot")

    plot_model(kmeans, plot='distribution', save=True)

    logging.info("6. Predict Model")

    pred_new = predict_model(kmeans, data=data)
    pred_new.head()

    logging.info("7. Save / Load Model")

    save_model(kmeans, model_name='kmeans')

    loaded_kmeans = load_model('kmeans')
    print(loaded_kmeans)

    logging.info("9. Get Config / Set Config")

    X = get_config('X')
    X.head()

    get_config('seed')

    set_config('seed', 999)

    get_config('seed')


if __name__ == "__main__":
    ex.run()
    for image in glob.glob("*.png"):
        ex.add_artifact(image)
        os.remove(image)
    for image in glob.glob("*.html"):
        ex.add_artifact(image)
        os.remove(image)
    for pkl in glob.glob("*.pkl"):
        ex.add_artifact(pkl)
        os.remove(pkl)
