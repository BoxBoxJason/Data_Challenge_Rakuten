import logging
from os import getenv, makedirs
from os.path import join
from sklearn.neighbors import RadiusNeighborsClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Radius Neighbors Classifier results directory path
__RADIUS_NEIGHBORS_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'radius_neighbors')

makedirs(__RADIUS_NEIGHBORS_RESULTS_PATH, exist_ok=True)


def optimizeRadiusNeighborsClassifierParameters(X_train, y_train):
    """
    @brief Trains a Radius Neighbors Classifier model on the training data.

    This function trains a Radius Neighbors Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Radius Neighbors Classifier model.
    """
    param_grid = {
        'radius': [1.0, 2.0, 3.0],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'outlier_label': [None],
        'p': [1, 2]
    }

    return optimizeModelParameters(RadiusNeighborsClassifier,'Radius Neighbors Classifier', param_grid, __RADIUS_NEIGHBORS_RESULTS_PATH, X_train,y_train)


def trainAndTestRadiusNeighborsClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Radius Neighbors Classifier model on the training data.

    This function trains a Radius Neighbors Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Radius Neighbors Classifier and the predictions.
    """
    logging.debug("Training Radius Neighbors Classifier")
    return trainAndTestModel(RadiusNeighborsClassifier, X_train, y_train, X_test, __RADIUS_NEIGHBORS_RESULTS_PATH)
