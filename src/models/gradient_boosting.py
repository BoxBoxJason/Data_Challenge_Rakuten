import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from utils.json import saveJson, convertToSerializable
from models.models import predictTestDataset, trainClassifier

# Gradient Boosting Classifier results path
GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'gradient_boosting_results')
# Best parameters for Gradient Boosting Classifier path
GRADIENT_BOOSTING_BEST_PARAMS_PATH = join(GRADIENT_BOOSTING_RESULTS_PATH, 'best_params.json')
# All tests results for Gradient Boosting Classifier path
GRADIENT_BOOSTING_ALL_TESTS_RESULTS_PATH = join(GRADIENT_BOOSTING_RESULTS_PATH, 'all_tests_results.json')
# Gradient Boosting predicted csv path
GRADIENT_BOOSTING_PREDICTED_CSV_PATH = join(GRADIENT_BOOSTING_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)

#Hist Gradient Boosting Classifier results path
HIST_GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'hist_gradient_boosting_results')
# Best parameters for Hist Gradient Boosting Classifier path
HIST_GRADIENT_BOOSTING_BEST_PARAMS_PATH = join(HIST_GRADIENT_BOOSTING_RESULTS_PATH, 'best_params.json')
# All tests results for Hist Gradient Boosting Classifier path
HIST_GRADIENT_BOOSTING_ALL_TESTS_RESULTS_PATH = join(HIST_GRADIENT_BOOSTING_RESULTS_PATH, 'all_tests_results.json')
# Hist Gradient Boosting predicted csv path
HIST_GRADIENT_BOOSTING_PREDICTED_CSV_PATH = join(HIST_GRADIENT_BOOSTING_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(HIST_GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)


def preProcessDatasetGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Gradient Boosting Classifier")
    return dataset

def preProcessDatasetHistGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Hist Gradient Boosting Classifier")
    return dataset


def optimizeGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Gradient Boosting Classifier.
    """
    logging.debug("Optimizing Gradient Boosting Classifier")
    classifier = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }
    grid_search = GridSearchCV(classifier, param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    best_params = grid_search.best_params_
    saveJson(GRADIENT_BOOSTING_BEST_PARAMS_PATH, convertToSerializable(best_params))
    return grid_search.best_estimator_


def optimizeHistGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Hist Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Hist Gradient Boosting Classifier.
    """
    logging.debug("Optimizing Hist Gradient Boosting Classifier")
    classifier = HistGradientBoostingClassifier()
    param_grid = {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [10, 20, 30]
    }
    grid_search = GridSearchCV(classifier, param_grid, cv=3)
    grid_search.fit(X_train, Y_train)
    best_params = grid_search.best_params_
    saveJson(HIST_GRADIENT_BOOSTING_BEST_PARAMS_PATH, convertToSerializable(best_params))
    return grid_search.best_estimator_

def trainGradientBoostingClassifier(X_train, Y_train):
    """
    @brief Trains the Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The trained Gradient Boosting Classifier.
    """
    logging.debug("Training Gradient Boosting Classifier")
    return trainClassifier(GradientBoostingClassifier(), X_train, Y_train)

def trainHistGradientBoostingClassifier(X_train, Y_train):
    """
    @brief Trains the Hist Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The trained Hist Gradient Boosting Classifier.
    """
    logging.debug("Training Hist Gradient Boosting Classifier")
    return trainClassifier(HistGradientBoostingClassifier(), X_train, Y_train)

def predictTestDatasetGradientBoosting(classifier, X_test):
    """
    @brief Predicts the test dataset with the Gradient Boosting Classifier.


    @param classifier The trained Gradient Boosting Classifier.
    @param X_test The test dataset.
    @return The predicted values for the test dataset.
    """
    logging.debug("Predicting test dataset with Gradient Boosting Classifier")
    return predictTestDataset(classifier, X_test)

def predictTestDatasetHistGradientBoosting(classifier, X_test):
    """
    @brief Predicts the test dataset with the Hist Gradient Boosting Classifier.


    @param classifier The trained Hist Gradient Boosting Classifier.
    @param X_test The test dataset.
    @return The predicted values for the test dataset.
    """
    logging.debug("Predicting test dataset with Hist Gradient Boosting Classifier")
    return predictTestDataset(classifier, X_test)

def saveResultsGradientBoosting(all_tests_results, Y_test):
    """
    @brief Saves the results for the Gradient Boosting Classifier.


    @param all_tests_results The results for all tests.
    @param Y_test The labels for the test dataset.
    """
    logging.debug("Saving results for Gradient Boosting Classifier")
    saveJson(GRADIENT_BOOSTING_ALL_TESTS_RESULTS_PATH, convertToSerializable(all_tests_results))
    Y_test.to_csv(GRADIENT_BOOSTING_PREDICTED_CSV_PATH, index=False)

def saveResultsHistGradientBoosting(all_tests_results, Y_test):
    """
    @brief Saves the results for the Hist Gradient Boosting Classifier.


    @param all_tests_results The results for all tests.
    @param Y_test The labels for the test dataset.
    """
    logging.debug("Saving results for Hist Gradient Boosting Classifier")
    saveJson(HIST_GRADIENT_BOOSTING_ALL_TESTS_RESULTS_PATH, convertToSerializable(all_tests_results))
    Y_test.to_csv(HIST_GRADIENT_BOOSTING_PREDICTED_CSV_PATH, index=False)