import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import PassiveAggressiveClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Passive Aggressive Classifier results directory path
__PASSIVE_AGGRESSIVE_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'passive_aggressive')

makedirs(__PASSIVE_AGGRESSIVE_RESULTS_PATH, exist_ok=True)


def optimizePassiveAggressiveParameters(X_train, y_train):
    """
    @brief Trains a Passive Aggressive Classifier model on the training data.

    This function trains a Passive Aggressive Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Passive Aggressive Classifier model.
    """
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'tol': [1e-3, 1e-4, 1e-5],
                  'loss': ['hinge', 'squared_hinge']
                  }


    return optimizeModelParameters(PassiveAggressiveClassifier, 'Passive Aggressive Classifier', param_grid, __PASSIVE_AGGRESSIVE_RESULTS_PATH, X_train, y_train)


def trainAndTestPassiveAggressiveModel(X_train, X_test, y_train):
    """
    @brief Trains and tests a Passive Aggressive Classifier model.

    This function trains a Passive Aggressive Classifier model on the training data and
    tests it on the testing data.

    @param X_train The training data.
    @param X_test The testing data.
    @param y_train The training target variable.
    @return None
    """
    logging.debug("Training Passive Aggressive Classifier model")
    trainAndTestModel(PassiveAggressiveClassifier, X_train, y_train, X_test, __PASSIVE_AGGRESSIVE_RESULTS_PATH)
