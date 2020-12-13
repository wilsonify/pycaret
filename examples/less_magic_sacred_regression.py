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
from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    models,
    plot_model,
    evaluate_model,
    interpret_model,
    predict_model,
    save_model,
    load_model,
    automl,
    tune_model,
    ensemble_model,
    blend_models,
    stack_models,
    get_config, set_config
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

ex = Experiment("pycaret regression on insurance dataset")
ex.observers.append(fso)
ex.observers.append(mdbo)
ex.add_config(
    dict(
        train_size=0.7,  # float
        test_data=None,  #: Optional[pd.DataFrame]
        preprocess=True,  # : bool = True,
        imputation_type="simple",  # : str = "simple",
        iterative_imputation_iters=5,  # : int = 5,
        categorical_features=None,  # : Optional[List[str]] = None,
        categorical_imputation="constant",  # : str = "constant",
        categorical_iterative_imputer="lightgbm",  # : Union[str, Any] = "lightgbm",
        ordinal_features=None,  # : Optional[Dict[str, list]] = None,
        high_cardinality_features=None,  # : Optional[List[str]] = None,
        high_cardinality_method="frequency",  # : str = "frequency",
        numeric_features=None,  # : Optional[List[str]] = None,
        numeric_imputation="mean",  # : str = "mean",
        numeric_iterative_imputer="lightgbm",  # : Union[str, Any] = "lightgbm",
        date_features=None,  # : Optional[List[str]] = None,
        ignore_features=None,  # : Optional[List[str]] = None,
        normalize=False,  # : bool = False,
        normalize_method="zscore",  # : str = "zscore",
        transformation=False,  # : bool = False,
        transformation_method="yeo-johnson",  # : str = "yeo-johnson",
        handle_unknown_categorical=True,  # : bool = True,
        unknown_categorical_method="least_frequent",  # : str = "least_frequent",
        pca=False,  # : bool = False,
        pca_method="linear",  # : str = "linear",
        pca_components=None,  # : Optional[float] = None,
        ignore_low_variance=False,  # : bool = False,
        combine_rare_levels=False,  # : bool = False,
        rare_level_threshold=0.10,  # : float = 0.10,
        bin_numeric_features=None,  # : Optional[List[str]] = None,
        remove_outliers=False,  # : bool = False,
        outliers_threshold=0.05,  # : float = 0.05,
        remove_multicollinearity=False,  # : bool = False,
        multicollinearity_threshold=0.9,  # : float = 0.9,
        remove_perfect_collinearity=True,  # : bool = True,
        create_clusters=False,  # : bool = False,
        cluster_iter=20,  # : int = 20,
        polynomial_features=False,  # : bool = False,
        polynomial_degree=2,  # : int = 2,
        trigonometry_features=False,  # : bool = False,
        polynomial_threshold=0.1,  # : float = 0.1,
        group_features=None,  # : Optional[List[str]] = None,
        group_names=None,  # : Optional[List[str]] = None,
        feature_selection=False,  # : bool = False,
        feature_selection_threshold=0.8,  # : float = 0.8,
        feature_selection_method="classic",  # : str = "classic",
        feature_interaction=False,  # : bool = False,
        feature_ratio=False,  # : bool = False,
        interaction_threshold=0.01,  # : float = 0.01,
        transform_target=False,  # : bool = False,
        transform_target_method="box-cox",  # : str = "box-cox",
        data_split_shuffle=True,  # : bool = True,
        data_split_stratify=False,  # : Union[bool, List[str]] = False,
        fold_strategy="kfold",  # : Union[str, Any] = "kfold",
        fold=10,  # : int = 10,
        fold_shuffle=False,  # : bool = False,
        fold_groups=None,  # : Optional[Union[str, pd.DataFrame]] = None,
        n_jobs=-1,  # : Optional[int] = -1,
        use_gpu=False,  # : bool = False,
        custom_pipeline=None,  # : Union[Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]] = None,
        html=True,  # : bool = True,
        session_id=None,  # : Optional[int] = None,
        log_experiment=False,  # : bool = False,
        experiment_name=None,  # : Optional[str] = None,
        log_plots=False,  # : Union[bool, list] = False,
        log_profile=False,  # : bool = False,
        log_data=False,  # : bool = False,
        silent=True,  # : bool = False,
        verbose=True,  # : bool = True,
        profile=False,  # : bool = False,
        profile_kwargs=None,  # : Dict[str, Any] = None,

    )
)


@ex.main
def run(_config):
    logging.info("1. Loading Dataset")

    from pycaret.datasets import get_data
    data = get_data('insurance')

    logging.info("2. Initialize Setup")

    reg1 = setup(
        data,
        target='charges',
        train_size=_config["train_size"],
        test_data=_config["test_data"],
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
        remove_outliers=_config["remove_outliers"],
        outliers_threshold=_config["outliers_threshold"],
        remove_multicollinearity=_config["remove_multicollinearity"],
        multicollinearity_threshold=_config["multicollinearity_threshold"],
        remove_perfect_collinearity=_config["remove_perfect_collinearity"],
        create_clusters=_config["create_clusters"],
        cluster_iter=_config["cluster_iter"],
        polynomial_features=_config["polynomial_features"],
        polynomial_degree=_config["polynomial_degree"],
        trigonometry_features=_config["trigonometry_features"],
        polynomial_threshold=_config["polynomial_threshold"],
        group_features=_config["group_features"],
        group_names=_config["group_names"],
        feature_selection=_config["feature_selection"],
        feature_selection_threshold=_config["feature_selection_threshold"],
        feature_selection_method=_config["feature_selection_method"],
        feature_interaction=_config["feature_interaction"],
        feature_ratio=_config["feature_ratio"],
        interaction_threshold=_config["interaction_threshold"],
        transform_target=_config["transform_target"],
        transform_target_method=_config["transform_target_method"],
        data_split_shuffle=_config["data_split_shuffle"],
        data_split_stratify=_config["data_split_stratify"],
        fold_strategy=_config["fold_strategy"],
        fold=_config["fold"],
        fold_shuffle=_config["fold_shuffle"],
        fold_groups=_config["fold_groups"],
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

    logging.info("3. Compare Baseline")

    best_model = compare_models(fold=5)
    top_five = compare_models(n_select=5, fold=5, include=list(models().index))

    logging.info("4. Create Model")

    lightgbm = create_model('lightgbm')

    import numpy as np
    lgbms = [create_model('lightgbm', learning_rate=i) for i in np.arange(0.1, 1, 0.1)]

    print(len(lgbms))

    logging.info("5. Tune Hyperparameters")

    tuned_lightgbm = tune_model(lightgbm, n_iter=50, optimize='MAE')

    tuned_lightgbm

    logging.info("6. Ensemble Model")

    dt = create_model('dt')

    bagged_dt = ensemble_model(dt, n_estimators=50)

    boosted_dt = ensemble_model(dt, method='Boosting')

    logging.info("7. Blend Models")

    blender = blend_models(top_five)

    logging.info("8. Stack Models")

    stacker = stack_models(
        estimator_list=top_five
    )

    logging.info("9. Analyze Model")

    plot_model(dt, plot="residuals", save=True)

    plot_model(dt, plot='error', save=True)

    plot_model(dt, plot='feature', save=True)

    evaluate_model(dt)

    logging.info("10. Interpret Model")

    interpret_model(lightgbm, plot="summary", show=False)
    plt.savefig("interpret lightgbm summary.png")

    interpret_model(lightgbm, plot='correlation', show=False)
    plt.savefig("interpret lightgbm correlation.png")

    interpret_model(lightgbm, plot='reason', observation=12, show=False)
    plt.savefig("interpret lightgbm reason for observation 12.png")

    logging.info("11. AutoML")

    best = automl(optimize='MAE')
    best

    logging.info("12. Predict Model")

    pred_holdouts = predict_model(lightgbm)
    pred_holdouts.head()

    new_data = data.copy()
    new_data.drop(['charges'], axis=1, inplace=True)
    predict_new = predict_model(best, data=new_data)
    predict_new.head()

    logging.info("13. Save / Load Model")

    save_model(best, model_name='best-model')

    loaded_bestmodel = load_model('best-model')
    print(loaded_bestmodel)

    logging.info("15. Get Config / Set Config")

    X_train = get_config('X_train')
    X_train.head()

    get_config('seed')

    set_config('seed', 999)

    get_config('seed')


if __name__ == "__main__":
    ex.run()
    for image in glob.glob("*.png"):
        ex.add_artifact(image)
        os.remove(image)
    for pkl in glob.glob("*.pkl"):
        ex.add_artifact(pkl)
        os.remove(pkl)
