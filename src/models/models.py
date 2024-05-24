import logging
from os.path import join
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from utils.json import saveJson, convertToSerializable

# Best params json file name
__BEST_PARAM_FILENAME = 'best_params.json'
# All results json file name
__ALL_RESULTS_FILENAME = 'all_results.json'


def predictTestDataset(model, processed_X_test, results_path=None):
    """
    @brief Predicts the target variable using the given model and test data.

    This function predicts the target variable using the given model and test data.
    The predictions are saved to a csv file if the results_path is provided.

    @param model The trained model.
    @param processed_X_test The test data.
    @param results_path The path to save the predictions.
    @return The predictions.
    """
    prediction = model.predict(processed_X_test)
    if results_path:
        prediction_df = DataFrame(prediction)
        prediction_df.to_csv(results_path, index=True)
    return prediction


def trainClassifier(model_class, model_params, processed_X_train, y_train):
    """
    @brief Trains a classifier model on the training data.

    This function trains a classifier model on the training data.

    @param model The classifier model.
    @param processed_X_train The training data.
    @param y_train The target variable.
    @return The trained classifier model.
    """
    model = model_class(**model_params, n_jobs=-1, verbose=2)
    model.fit(processed_X_train, y_train)
    return model


def optimizeModelParameters(model_class, model_name, model_params_grid, model_results_path, processed_X_train, y_train):
    logging.debug(f"Optimizing {model_name} hyperparameters")
    model = model_class()

    grid_search = GridSearchCV(estimator=model, param_grid=model_params_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for {model_name} : {grid_search.best_params_}")
    logging.info(f"Best score for {model_name}: {grid_search.best_score_}")

    best_params_path = join(model_results_path,__BEST_PARAM_FILENAME)
    logging.info(f"Saving best parameters at {best_params_path}")
    saveJson(grid_search.best_params_, best_params_path)

    all_tests_results_path = join(model_results_path, __ALL_RESULTS_FILENAME)
    logging.info(f"Saving all tests results at {all_tests_results_path}")
    saveJson(convertToSerializable(grid_search.cv_results_), all_tests_results_path)

    return grid_search.best_estimator_
