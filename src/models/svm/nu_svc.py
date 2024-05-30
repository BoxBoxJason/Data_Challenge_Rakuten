import logging
from os import getenv, makedirs
from os.path import join
from sklearn.svm import NuSVC
from models.models import trainAndTestModel, optimizeModelParameters

# NuSVC Classifier results directory path
__NU_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'nu_svc')

makedirs(__NU_SVC_RESULTS_PATH, exist_ok=True)


def optimizeNuSVCParameters(X_train, y_train):
    """
    @brief Trains a NuSVC Classifier model on the training data.

    This function trains a NuSVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained NuSVC Classifier model.
    """
    param_grid = {
        'nu': [0.1, 0.5, 0.9],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
    }

    return optimizeModelParameters(NuSVC,'NuSVC Classifier', param_grid, __NU_SVC_RESULTS_PATH, X_train,y_train)


def trainAndTestNuSVC(X_train, y_train, X_test):
    """
    @brief Trains a NuSVC Classifier model on the training data.

    This function trains a NuSVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained NuSVC Classifier and the predictions.
    """
    logging.info("Training NuSVC model")
    return trainAndTestModel(NuSVC, X_train, y_train, X_test, __NU_SVC_RESULTS_PATH)
