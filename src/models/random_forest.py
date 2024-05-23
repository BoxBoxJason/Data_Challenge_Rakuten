import logging
from os.path import join
from os import getenv, makedirs
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from utils.json import saveJson

# Random Forest Regressor results path
__RANDOM_FOREST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'random_forest_results')
# Best parameters for Random Forest Regressor path
__RANDOM_FOREST_BEST_PARAMS_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'best_params.json')
# All tests results for Random Forest Regressor path
__RANDOM_FOREST_ALL_TESTS_RESULTS_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'all_tests_results.json')

makedirs(__RANDOM_FOREST_RESULTS_PATH, exist_ok=True)


def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Random Forest Regressor")
    return dataset


def optimizeRandomForestRegressorParameters(X_train, y_train):
    """
    @brief Trains a Random Forest Regressor model on the training data.

    This function trains a Random Forest Regressor model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Random Forest Regressor model.
    """
    logging.debug("Training Random Forest Regressor model")
    rf = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300,400,500],
        'max_depth': [None,10, 20, 30],
        'min_samples_split': [2, 5, 10,15],
        'min_samples_leaf': [1, 2, 4,8],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    print(f'y_train shape: {y_train.shape}')

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    processed_X_train = preProcessDataset(X_train)
    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for Random Forest Regressor: {grid_search.best_params_}")
    logging.info(f"Best score for Random Forest Regressor: {grid_search.best_score_}")

    logging.info(f"Saving best parameters at {__RANDOM_FOREST_BEST_PARAMS_PATH}")
    saveJson(grid_search.best_params_, __RANDOM_FOREST_BEST_PARAMS_PATH)

    logging.info(f"Saving all tests results at {__RANDOM_FOREST_ALL_TESTS_RESULTS_PATH}")
    saveJson(grid_search.cv_results_, __RANDOM_FOREST_ALL_TESTS_RESULTS_PATH)

    return grid_search.best_estimator_
