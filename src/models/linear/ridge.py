import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import RidgeClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Ridge Classifier results directory path
__RIDGE_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'ridge')

makedirs(__RIDGE_RESULTS_PATH, exist_ok=True)


def optimizeRidgeParameters(X_train, y_train):
    """
    @brief Trains a Ridge Classifier model on the training data.

    This function trains a Ridge Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Ridge Classifier model.
    """
    param_grid = {'alpha': [0.1, 1, 10, 100, 1000]}


    return optimizeModelParameters(RidgeClassifier, 'Ridge Classifier', param_grid, __RIDGE_RESULTS_PATH, X_train, y_train)


def trainAndTestRidgeModel(X_train, X_test, y_train):
    """
    @brief Trains and tests a Ridge Classifier model.

    This function trains a Ridge Classifier model on the training data and
    tests it on the testing data.

    @param X_train The training data.
    @param X_test The testing data.
    @param y_train The training target variable.
    @param y_test The testing target variable.
    """
    logging.debug("Training Ridge Classifier model")
    trainAndTestModel(RidgeClassifier, X_train, y_train, X_test, __RIDGE_RESULTS_PATH)
