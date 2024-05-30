import logging
from os.path import join
from os import getenv, makedirs
from sklearn.neighbors import KNeighborsClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# K Neighbors Classifier results path
__K_NEIGHBORS_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'k_nearest_neighbors')

makedirs(__K_NEIGHBORS_RESULTS_PATH, exist_ok=True)


def optimizeKNeighborsClassifierParameters(X_train, y_train):
    """
    @brief Trains a K Neighbors Classifier model on the training data.

    This function trains a K Neighbors Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained K Neighbors Classifier model.
    """
    param_grid = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    }

    return optimizeModelParameters(KNeighborsClassifier,'K-Nearest Neighbors Classifier', param_grid,__K_NEIGHBORS_RESULTS_PATH,X_train,y_train)


def trainAndTestKNeighborsClassifier(X_train, y_train, X_test):
    """
    @brief Trains a K Neighbors Classifier model on the training data.

    This function trains a K Neighbors Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained K Neighbors Classifier and the predictions.
    """
    logging.info("Training K Neighbors Classifier")
    return trainAndTestModel(KNeighborsClassifier, X_train, y_train, X_test, __K_NEIGHBORS_RESULTS_PATH)