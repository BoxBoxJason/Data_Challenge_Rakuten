import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import ElasticNet
from models.models import trainAndTestModel, optimizeModelParameters

# Elastic Net results directory path
__ELASTIC_NET_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'elastic_net')

makedirs(__ELASTIC_NET_RESULTS_PATH, exist_ok=True)


def optimizeElasticNetParameters(X_train, y_train):
    """
    @brief Trains an Elastic Net model on the training data.

    This function trains an Elastic Net model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Elastic Net model.
    """
    param_grid = {
        'alpha': [0.0001, 0.001, 0.01,1,10],
        'l1_ratio': [0.15, 0.25, 0.35],
        'tol': [0.0001, 0.001, 0.01],
        'selection': ['cyclic', 'random'],
    }

    return optimizeModelParameters(ElasticNet, 'Elastic Net', param_grid, __ELASTIC_NET_RESULTS_PATH, X_train, y_train)


def trainAndTestElasticNet(X_train, y_train, X_test):
    """
    @brief Trains an Elastic Net model on the training data.

    This function trains an Elastic Net model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Elastic Net and the predictions.
    """
    logging.debug("Training Elastic Net")
    return trainAndTestModel(ElasticNet, X_train, y_train, X_test, __ELASTIC_NET_RESULTS_PATH)
