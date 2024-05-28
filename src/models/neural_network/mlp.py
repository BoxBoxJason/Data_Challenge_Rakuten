import logging
from os.path import join
from os import getenv, makedirs
from sklearn.neural_network import MLPClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# MLP Classifier results path
__MLP_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'mlp')

makedirs(__MLP_RESULTS_PATH, exist_ok=True)


def optimizeMLPClassifierParameters(X_train, y_train):
    """
    @brief Trains a MLP Classifier model on the training data.

    This function trains a MLP Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained MLP Classifier model.
    """
    param_grid = [
        {}
    ]

    return optimizeModelParameters(MLPClassifier, 'MLP Classifier', param_grid, __MLP_RESULTS_PATH, X_train, y_train)


def trainAndTestMLPClassifier(X_train, y_train, X_test):
    """
    @brief Trains a MLP Classifier model on the training data.

    This function trains a MLP Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained MLP Classifier and the predictions.
    """
    logging.debug("Training MLP Classifier")
    return trainAndTestModel(MLPClassifier, X_train, y_train, X_test, __MLP_RESULTS_PATH)
