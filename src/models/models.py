import logging
from os.path import join
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
from utils.json import saveJson, convertToSerializable, loadJson

# Best params json file name
__BEST_PARAM_FILENAME = 'best_params.json'
# All results json file name
__ALL_RESULTS_FILENAME = 'all_results.json'
# Predicted csv file name
__PREDICTED_CSV_FILENAME = 'predicted_Y_test.csv'

def trainAndTestModel(model_class, processed_X_train, y_train, processed_X_test, results_path=None):
    """
    @brief Trains a model on the training data and predicts the target variable using the test data.

    This function trains a model on the training data and predicts the target variable using the test data.
    The predictions are saved to a csv file if the results_path is provided.

    @param model The model to be trained.
    @param processed_X_train The training data.
    @param y_train The target variable.
    @param processed_X_test The test data.
    @param results_path The path to save the predictions.
    @return The trained model and the predictions.
    """
    model_best_params = getModelBestParams(results_path)
    model = trainClassifier(model_class, model_best_params, processed_X_train, y_train)
    return model, predictTestDataset(model, processed_X_test, results_path)


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
        prediction_path = join(results_path, __PREDICTED_CSV_FILENAME)
        logging.info(f"Saving predictions at {prediction_path}")

        # Set the column name to the target variable name and add offset to the index
        prediction_df = DataFrame(prediction)
        prediction_df.columns = ['prdtypecode']

        prediction_df.index += 84916

        prediction_df.to_csv(prediction_path, index=True)
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
    model = model_class(**model_params)
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


def getModelBestParams(model_results_path):
    """
    @brief Returns the best parameters path for the given model results path.


    @param model_results_path The path to the model results.
    @return The best parameters json object.
    """
    best_params_path = join(model_results_path, __BEST_PARAM_FILENAME)
    best_params = {}
    try:
        best_params = loadJson(best_params_path)
    except FileNotFoundError:
        logging.error(f"Best parameters file not found at {best_params_path}, using default parameters.")
    return best_params
