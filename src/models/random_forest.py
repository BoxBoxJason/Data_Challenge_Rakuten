import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
from utils.json import saveJson

# Random Forest Regressor results path
__RANDOM_FOREST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'random_forest_results')
# Best parameters for Random Forest Regressor path
__RANDOM_FOREST_BEST_PARAMS_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'best_params.json')
# All tests results for Random Forest Regressor path
__RANDOM_FOREST_ALL_TESTS_RESULTS_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'all_tests_results.json')
# Random forest predicted csv path
__RANDOM_FOREST_PREDICTED_CSV_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'predicted_Y_test.csv')

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
        'max_features': ['auto', 'sqrt', 'log2']
    }

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


def trainRandomForestRegressor(X_train, y_train, best_params=None, n_estimators=100, max_depth=None, max_features='auto'):
    """
    @brief Trains a Random Forest Regressor model on the training data.

    This function trains a Random Forest Regressor model on the training data.
    The model is trained using the best hyperparameters found by optimizeRandomForestRegressorParameters().

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Random Forest Regressor model.
    """
    logging.debug("Training Random Forest Regressor model")
    if best_params is not None:
        rf = RandomForestClassifier(**best_params, n_jobs=-1, verbose=2, random_state=42)
    else:
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                    n_jobs=-1, verbose=2, random_state=42)
    processed_X_train = preProcessDataset(X_train)
    rf.fit(processed_X_train, y_train)
    return rf


def predictTestDataset(rf, X_test, save_results=False):
    """
    @brief Predicts the target variable using a trained Random Forest Regressor model.

    This function predicts the target variable using a trained Random Forest Regressor model.

    @param rf The trained Random Forest Regressor model.
    @param X_test The test data.
    @return The predicted target variable.
    """
    processed_X_test = preProcessDataset(X_test)
    prediction = rf.predict(processed_X_test)
    if save_results:
        prediction_df = DataFrame(prediction)
        prediction_df.to_csv(__RANDOM_FOREST_PREDICTED_CSV_PATH, index=True)
    return prediction
