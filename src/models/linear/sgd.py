import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import SGDClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Stochastic Gradient Descent Classifier results directory path
__SGD_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'sgd')

makedirs(__SGD_RESULTS_PATH, exist_ok=True)


def optimizeSGDParameters(X_train, y_train):
    """
    @brief Trains a Stochastic Gradient Descent Classifier model on the training data.

    This function trains a Stochastic Gradient Descent Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Stochastic Gradient Descent Classifier model.
    """
    param_grid = {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                  'penalty': ['l2', 'l1', 'elasticnet'],
                  'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                  'l1_ratio': [0.15, 0.25, 0.5, 0.75, 0.9],
                  'tol': [1e-3, 1e-4, 1e-5]
                  }

    return optimizeModelParameters(SGDClassifier, 'Stochastic Gradient Descent Classifier', param_grid, __SGD_RESULTS_PATH, X_train, y_train)


def trainAndTestSGDModel(X_train, X_test, y_train):
    """
    @brief Trains and tests a Stochastic Gradient Descent Classifier model.

    This function trains a Stochastic Gradient Descent Classifier model on the training data and
    tests it on the testing data.

    @param X_train The training data.
    @param X_test The testing data.
    @param y_train The training target variable.
    @param y_test The testing target variable.
    """
    logging.debug("Training Stochastic Gradient Descent Classifier model")
    trainAndTestModel(SGDClassifier, X_train, y_train, X_test, __SGD_RESULTS_PATH)
