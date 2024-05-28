import logging
import inspect
from os.path import join, isdir
from os import listdir, getenv
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from utils.json import saveJson, convertToSerializable, loadJson
from graphs.graphs import drawScoresBarChart

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
    init_signature = inspect.signature(model_class.__init__)

    if 'n_jobs' in init_signature.parameters:
        model_params.setdefault('n_jobs', -1)
    model = model_class(**model_params)
    model.fit(processed_X_train, y_train)
    return model


def optimizeModelParameters(model_class, model_name, model_params_grid, model_results_path, processed_X_train, y_train):
    model = model_class()

    best_params = {}
    best_score = 0
    all_results = {}

    if isinstance(model_params_grid, dict):
        logging.debug(f"Optimizing {model_name} hyperparameters with GridSearchCV")

        grid_search = GridSearchCV(estimator=model, param_grid=model_params_grid, cv=5, n_jobs=-1, verbose=3, scoring='f1_weighted')
        grid_search.fit(processed_X_train, y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        all_results = convertToSerializable(grid_search.cv_results_)

    # Case where the GridsearchCV does not suit because they did not optimize it
    elif isinstance(model_params_grid, list):
        logging.debug(f"Optimizing {model_name} hyperparameters manually")
        best_score, best_params, all_results = gridSearchCV(model_name, model_class, model_params_grid, processed_X_train, y_train)

    logging.info(f"Best parameters for {model_name} : {best_params}")
    logging.info(f"Best score for {model_name}: {best_score}")

    best_params_path = join(model_results_path,__BEST_PARAM_FILENAME)
    logging.info(f"Saving best parameters at {best_params_path}")
    saveJson(best_params, best_params_path)

    all_tests_results_path = join(model_results_path, __ALL_RESULTS_FILENAME)
    logging.info(f"Saving all tests results at {all_tests_results_path}")
    saveJson(all_results, all_tests_results_path)


def gridSearchCV(model_name,model_class, model_params_grid, processed_X_train, y_train, cv=5):
    # Build the results dictionary manually
    all_results = {
        "params" : model_params_grid,
        "mean_test_score" : [],
        "std_test_score": []
    }
    for i in range(cv):
        all_results[f"split{i}_test_score"] = []
    best_score = 0
    best_params = {}

    init_signature = inspect.signature(model_class.__init__)
    for params in model_params_grid:
        if 'n_jobs' in init_signature.parameters:
            params.setdefault('n_jobs', -1)

        cv_scores = []
        for i in range(cv):
            logging.debug(f"Training {model_name} validation {i+1}/{cv} with parameters {params}")
            model_instance = model_class(**params)
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(processed_X_train, y_train, test_size=0.2, random_state=42)
            model_instance.fit(X_train_split, y_train_split)
            y_val_pred = model_instance.predict(X_val_split)
            score = f1_score(y_val_split, y_val_pred, average='weighted')
            logging.info(f"Score for {model_name} validation {i+1}/{cv}, with parameters {params} : {score}")
            all_results[f"split{i}_test_score"].append(score)
            cv_scores.append(score if not str(score) == 'nan' else 0)

        mean_tests_score = sum(cv_scores) / len(cv_scores)
        tests_std = sum([(score - mean_tests_score) ** 2 for score in cv_scores]) / len(cv_scores)

        if mean_tests_score > best_score:
            best_score = score
            best_params = params

        all_results["mean_test_score"].append(mean_tests_score)
        all_results["std_test_score"].append(tests_std)

    return best_score, best_params, all_results


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


def drawValidationScores(show=True):
    """
    @brief Draws scores comparison graph for all models.

    This function draws scores comparison graph for all models.
    """

    result_path = getenv('PROJECT_RESULTS_DIR')
    scores = {}
    for file in listdir(result_path):
        file_path = join(result_path, file)
        if isdir(file_path):
            all_results_path = join(file_path, __ALL_RESULTS_FILENAME)
            try:
                all_results = loadJson(all_results_path)
                scores[file] = max([score for score in all_results['mean_test_score'] if str(score) != 'nan'])
            except FileNotFoundError:
                logging.error(f"File not found at {all_results_path}")
    drawScoresBarChart('Weighted F1', scores, join(result_path, 'validation_scores.png'),show)

    return scores


def drawRealScores(show=True):
    """
    @brief Draws scores comparison graph for all models.

    This function draws scores comparison graph for all models.
    """

    result_path = join(getenv('PROJECT_RESULTS_DIR'),'weighted_f1_scores.json')
    real_scores = loadJson(result_path)

    drawScoresBarChart('Weighted F1', real_scores, join(getenv('PROJECT_RESULTS_DIR'), 'real_scores.png'),show)

    return real_scores
