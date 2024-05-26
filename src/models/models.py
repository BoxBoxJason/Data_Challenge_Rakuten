import logging
import inspect
from os.path import join, isdir
from os import listdir, getenv
from pandas import DataFrame
from sklearn.model_selection import GridSearchCV
import seaborn
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
    logging.debug(f"Optimizing {model_name} hyperparameters")
    model = model_class()

    grid_search = GridSearchCV(estimator=model, param_grid=model_params_grid, cv=5, n_jobs=-1, verbose=3, scoring='f1_weighted')

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


def drawGraphs(model_name, results_path, x_feature, hue_feature=None, col_feature=None, row_feature=None, style_feature=None):
    """
    @brief Draws graphs for the given results.

    This function draws graphs for the given results.
    Display mean test scores for each hyperparameter which are max_depth, max_features and n_estimators.

    @param results The results to draw graphs for.
    @param model_name The model name.
    @param results_path The path to save the graphs.
    """
    all_results_path = join(results_path, __ALL_RESULTS_FILENAME)
    all_results = loadJson(all_results_path)
    logging.info(f"Drawing graphs for {model_name} results")

    params_test_list = all_results['params']
    test_scores = all_results['mean_test_score']

    df = DataFrame(params_test_list)
    df['mean_test_score'] = test_scores
    df.fillna(0, inplace=True)

    seaborn.set_theme(style="whitegrid")
    graph = seaborn.catplot(data=df, x=x_feature, y='mean_test_score', hue=hue_feature, palette='mako',col=col_feature, kind='bar')
    graph.savefig(join(results_path, f'{model_name}_mean_test_scores.png'))


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
