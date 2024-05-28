import logging
from os import getenv, makedirs
from os.path import join
from sklearn.svm import SVC
from models.models import trainAndTestModel, optimizeModelParameters

# SVC Classifier results directory path
__SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'svc')

makedirs(__SVC_RESULTS_PATH, exist_ok=True)


def optimizeSVCParameters(X_train, y_train):
    """
    @brief Trains a SVC Classifier model on the training data.

    This function trains a SVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained SVC Classifier model.
    """
    param_grid = param_grid = [
        {'C': 0.1, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 0.1, 'kernel': 'rbf', 'gamma': 'auto'},
        {'C': 1, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 1, 'kernel': 'rbf', 'gamma': 'auto'},
        {'C': 10, 'kernel': 'rbf', 'gamma': 'scale'},
        {'C': 10, 'kernel': 'rbf', 'gamma': 'auto'},
        {'C': 0.1, 'kernel': 'poly', 'degree': 2, 'gamma': 'scale'},
        {'C': 0.1, 'kernel': 'poly', 'degree': 2, 'gamma': 'auto'},
        {'C': 0.1, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
        {'C': 0.1, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto'},
        {'C': 1, 'kernel': 'poly', 'degree': 2, 'gamma': 'scale'},
        {'C': 1, 'kernel': 'poly', 'degree': 2, 'gamma': 'auto'},
        {'C': 1, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
        {'C': 1, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto'},
        {'C': 10, 'kernel': 'poly', 'degree': 2, 'gamma': 'scale'},
        {'C': 10, 'kernel': 'poly', 'degree': 2, 'gamma': 'auto'},
        {'C': 10, 'kernel': 'poly', 'degree': 3, 'gamma': 'scale'},
        {'C': 10, 'kernel': 'poly', 'degree': 3, 'gamma': 'auto'},
        {'C': 0.1, 'kernel': 'sigmoid', 'gamma': 'scale'},
        {'C': 0.1, 'kernel': 'sigmoid', 'gamma': 'auto'},
        {'C': 1, 'kernel': 'sigmoid', 'gamma': 'scale'},
        {'C': 1, 'kernel': 'sigmoid', 'gamma': 'auto'},
        {'C': 10, 'kernel': 'sigmoid', 'gamma': 'scale'},
        {'C': 10, 'kernel': 'sigmoid', 'gamma': 'auto'}
    ]

    return optimizeModelParameters(SVC,'SVC', param_grid, __SVC_RESULTS_PATH, X_train,y_train)


def trainAndTestSVC(X_train, y_train, X_test):
    """
    @brief Trains a SVC Classifier model on the training data.

    This function trains a SVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained SVC Classifier and the predictions.
    """
    logging.debug("Training SVC model")
    return trainAndTestModel(SVC, X_train, y_train, X_test, __SVC_RESULTS_PATH)
