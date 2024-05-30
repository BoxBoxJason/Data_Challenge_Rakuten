import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import Perceptron
from models.models import trainAndTestModel, optimizeModelParameters

# Perceptron results directory path
__PERCEPTRON_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'perceptron')

makedirs(__PERCEPTRON_RESULTS_PATH, exist_ok=True)


def optimizePerceptronParameters(X_train, y_train):
    """
    @brief Trains a Perceptron model on the training data.

    This function trains a Perceptron model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Perceptron model.
    """
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.0001, 0.001, 0.01],
        'l1_ratio': [0.15, 0.25, 0.35],
        'tol': [0.0001, 0.001, 0.01],
        'eta0': [1.0, 10.0, 100.0],
    }

    return optimizeModelParameters(Perceptron, 'Perceptron', param_grid, __PERCEPTRON_RESULTS_PATH, X_train, y_train)


def trainAndTestPerceptron(X_train, y_train, X_test):
    """
    @brief Trains a Perceptron model on the training data.

    This function trains a Perceptron model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Perceptron and the predictions.
    """
    logging.info("Training Perceptron")
    return trainAndTestModel(Perceptron, X_train, y_train, X_test, __PERCEPTRON_RESULTS_PATH)
