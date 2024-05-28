import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from models.models import trainAndTestModel, optimizeModelParameters, drawGraphs

# Gradient Boosting Classifier results path
__GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'gradient_boosting')

#Hist Gradient Boosting Classifier results path
__HIST_GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'hist_gradient_boosting')

makedirs(__GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)
makedirs(__HIST_GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)


def preProcessDatasetGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Gradient Boosting Classifier")
    return dataset


def optimizeGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Gradient Boosting Classifier.
    """
    param_grid = [
        {
            'n_estimators': 300,
            'learning_rate': 0.1
        },
        {
            'n_estimators': 300,
            'learning_rate': 0.01
        },
        {
            'n_estimators': 300,
            'learning_rate': 1
        },
        {
            'n_estimators': 300,
            'learning_rate': 10
        }
    ]

    processed_X_train = preProcessDatasetGradientBoosting(X_train)

    return optimizeModelParameters(GradientBoostingClassifier,'Gradient Boosting Classifier',param_grid, __GRADIENT_BOOSTING_RESULTS_PATH, processed_X_train, Y_train)


def trainAndTestGradientBoostingClassifier(X_train, Y_train, X_test):
    """
    @brief Trains a Gradient Boosting Classifier model on the training data.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Gradient Boosting Classifier and the predictions.
    """
    logging.debug("Training Gradient Boosting Classifier")

    processed_X_train = preProcessDatasetGradientBoosting(X_train)
    processed_X_test = preProcessDatasetGradientBoosting(X_test)

    return trainAndTestModel(GradientBoostingClassifier, processed_X_train, Y_train, processed_X_test, __GRADIENT_BOOSTING_RESULTS_PATH)


def preProcessDatasetHistGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Hist Gradient Boosting Classifier")
    return dataset



def optimizeHistGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Hist Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Hist Gradient Boosting Classifier.
    """
    param_grid = {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [10, 20, 30]
    }
    processed_X_train = preProcessDatasetHistGradientBoosting(X_train)

    return optimizeModelParameters(HistGradientBoostingClassifier,'Hist Gradient Boosting Classifier',param_grid, __HIST_GRADIENT_BOOSTING_RESULTS_PATH, processed_X_train, Y_train)


def trainAndTestHistGradientBoostingClassifier(X_train, Y_train, X_test):
    """
    @brief Trains a Hist Gradient Boosting Classifier model on the training data.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Hist Gradient Boosting Classifier and the predictions.
    """
    logging.debug("Training Hist Gradient Boosting Classifier")

    processed_X_train = preProcessDatasetHistGradientBoosting(X_train)
    processed_X_test = preProcessDatasetHistGradientBoosting(X_test)

    return trainAndTestModel(HistGradientBoostingClassifier, processed_X_train, Y_train, processed_X_test, __HIST_GRADIENT_BOOSTING_RESULTS_PATH)


def drawGraphsGradientBoosting():
    """
    @brief Draws graphs for Gradient Boosting Classifier.

    This function draws graphs for Gradient Boosting Classifier.
    """
    drawGraphs('Gradient Boosting Classifier', __GRADIENT_BOOSTING_RESULTS_PATH)
