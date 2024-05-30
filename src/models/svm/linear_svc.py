import logging
from os import getenv, makedirs
from os.path import join
from sklearn.svm import LinearSVC
from models.models import trainAndTestModel, optimizeModelParameters

# LinearSVC Classifier results directory path
__LINEAR_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'linear_svc')

makedirs(__LINEAR_SVC_RESULTS_PATH, exist_ok=True)


def optimizeLinearSVCParameters(X_train, y_train):
    """
    @brief Trains a Linear SVC Classifier model on the training data.

    This function trains a Linear SVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Linear SVC Classifier model.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'penalty': ['l1', 'l2'],
        'multi_class': ['ovr', 'crammer_singer'],
    }

    return optimizeModelParameters(LinearSVC,'Linear SVC Classifier', param_grid, __LINEAR_SVC_RESULTS_PATH, X_train,y_train)


def trainAndTestLinearSVC(X_train, y_train, X_test):
    """
    @brief Trains a Linear SVC Classifier model on the training data.

    This function trains a Linear SVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Linear SVC Classifier and the predictions.
    """
    logging.info("Training Linear SVC model")
    return trainAndTestModel(LinearSVC, X_train, y_train, X_test, __LINEAR_SVC_RESULTS_PATH)
