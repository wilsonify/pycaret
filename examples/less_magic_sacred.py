"""A standard machine learning task without much sacred magic."""
import logging

import sklearn
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver, FileStorageObserver
from sacred.observers.file_storage import DEFAULT_FILE_STORAGE_PRIORITY
from sacred.observers.mongo import DEFAULT_MONGO_PRIORITY
from pycaret.datasets import get_data
from pycaret.classification import (
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

ex = Experiment("pycaret classification on juice1 dataset")
ex.observers.append(fso)
ex.observers.append(mdbo)
ex.add_config(
    dict(
        train_size=0.7,  #: float = 0.7,
        test_data=None,  #: Optional[pd.DataFrame] = None,
        preprocess=True,  #: bool = True,
        imputation_type="simple",  #: str = "simple",
        iterative_imputation_iters=5,  #: int = 5,
        categorical_features=None,  #: Optional[List[str]] = None,
        categorical_imputation="constant",  #: str = "constant",
        categorical_iterative_imputer="lightgbm",  #: Union[str, Any] = "lightgbm",
        ordinal_features=None,  #: Optional[Dict[str, list]] = None,
        high_cardinality_features=None,  #: Optional[List[str]] = None,
        high_cardinality_method="frequency",  #: str = "frequency",
        numeric_features=None,  #: Optional[List[str]] = None,
        numeric_imputation="mean",  #: str = "mean",
        numeric_iterative_imputer="lightgbm",  #: Union[str, Any] = "lightgbm",
        date_features=None,  #: Optional[List[str]] = None,
        ignore_features=None,  #: Optional[List[str]] = None,
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
        remove_outliers=False,  #: bool = False,
        outliers_threshold=0.05,  #: float = 0.05,
        remove_multicollinearity=False,  #: bool = False,
        multicollinearity_threshold=0.9,  #: float = 0.9,
        remove_perfect_collinearity=True,  #: bool = True,
        create_clusters=False,  #: bool = False,
        cluster_iter=20,  #: int = 20,
        polynomial_features=False,  #: bool = False,
        polynomial_degree=2,  #: int = 2,
        trigonometry_features=False,  #: bool = False,
        polynomial_threshold=0.1,  #: float = 0.1,
        group_features=None,  #: Optional[List[str]] = None,
        group_names=None,  #: Optional[List[str]] = None,
        feature_selection=False,  #: bool = False,
        feature_selection_threshold=0.8,  #: float = 0.8,
        feature_selection_method="classic",  #: str = "classic",
        feature_interaction=False,  #: bool = False,
        feature_ratio=False,  #: bool = False,
        interaction_threshold=0.01,  #: float = 0.01,
        fix_imbalance=False,  #: bool = False,
        fix_imbalance_method=None,  #: Optional[Any] = None,
        data_split_shuffle=True,  #: bool = True,
        data_split_stratify=False,  #: Union[bool, List[str]] = False,
        fold_strategy="stratifiedkfold",  #: Union[str, Any] = "stratifiedkfold",
        fold=10,  #: int = 10,
        fold_shuffle=False,  #: bool = False,
        fold_groups=None,  #: Optional[Union[str, pd.DataFrame]] = None,
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
        silent=True,  #: bool = False, disables prompt about datatypes
        verbose=True,  #: bool = True,
        profile=False,  #: bool = False,
        profile_kwargs=None,  #: Dict[str, Any] = None,

    )
)


@ex.main
def run(_config):
    """
    # Using main, command-line arguments will not be interpreted in any special way.

    Args:
        _config: sacred config

    Returns:
        performance of best model

    """
    logging.info("start run")
    index = get_data('index')
    data = get_data('juice')

    clf1 = setup(
        data=data,
        target='Purchase',
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
        fix_imbalance=_config["fix_imbalance"],
        fix_imbalance_method=_config["fix_imbalance_method"],
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

    best_model = compare_models()
    lr = create_model('lr')
    dt = create_model('dt')
    rf = create_model('rf', fold=5)

    ensembled_models = compare_models(
        include=[
            'rf',
            'ada',
            'gbc',
            'et',
            'xgboost',
            'lightgbm',
            'catboost'
        ], fold=3)
    tuned_lr = tune_model(lr)
    tuned_rf = tune_model(rf)

    bagged_dt = ensemble_model(dt)
    boosted_dt = ensemble_model(dt, method='Boosting')
    blender = blend_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], method='soft')
    stacker = stack_models(estimator_list=[boosted_dt, bagged_dt, tuned_rf], meta_model=rf)

    # # 9. Analyze Model
    plot_model(rf)
    plot_model(rf, plot='confusion_matrix')
    plot_model(rf, plot='boundary')
    plot_model(rf, plot='feature')
    plot_model(rf, plot='pr')
    plot_model(rf, plot='class_report')
    evaluate_model(rf)

    catboost = create_model('catboost', cross_validation=False)
    interpret_model(catboost)
    interpret_model(catboost, plot='correlation')
    interpret_model(catboost, plot='reason', observation=12)

    best = automl(optimize='Recall')
    print(best)

    pred_holdouts = predict_model(lr)
    print(pred_holdouts.head())

    new_data = data.copy()
    new_data.drop(['Purchase'], axis=1, inplace=True)
    predict_new = predict_model(best, data=new_data)
    predict_new.head()

    save_model(best, model_name='best-model')
    loaded_bestmodel = load_model('best-model')
    print(loaded_bestmodel)

    sklearn.set_config(display='diagram')
    loaded_bestmodel[0]
    sklearn.set_config(display='text')

    # deploy_model(best, model_name='best-aws', authentication={'bucket': 'pycaret-test'})

    X_train = get_config('X_train')
    X_train.head()

    get_config('seed')
    set_config('seed', 999)
    get_config('seed')
    evaluate_model(rf)
    return pred_holdouts


if __name__ == "__main__":
    ex.run()
