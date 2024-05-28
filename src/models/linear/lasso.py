import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import Lasso
from models.models import trainAndTestModel, optimizeModelParameters

# Lasso results directory path
__LASSO_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'lasso')

makedirs(__LASSO_RESULTS_PATH, exist_ok=True)


def optimizeLassoParameters(X_train, y_train):
    """
    @brief Trains a Lasso model on the training data.

    This function trains a Lasso model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Lasso model.
    """
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01,1,10],
        'tol': [0.0001, 0.001, 0.01],
        'selection': ['cyclic', 'random'],
    }

    return optimizeModelParameters(Lasso, 'Lasso', param_grid, __LASSO_RESULTS_PATH, X_train, y_train)


def trainAndTestLasso(X_train, y_train, X_test):
    """
    @brief Trains a Lasso model on the training data.

    This function trains a Lasso model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Lasso and the predictions.
    """
    logging.debug("Training Lasso")
    return trainAndTestModel(Lasso, X_train, y_train, X_test, __LASSO_RESULTS_PATH)
