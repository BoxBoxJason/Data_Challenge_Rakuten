import logging
from os import getenv, makedirs
from os.path import join
from sklearn.linear_model import LogisticRegression
from models.models import trainAndTestModel, optimizeModelParameters

# Logistic Regression results directory path
__LOGISTIC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'logistic_regression')

makedirs(__LOGISTIC_RESULTS_PATH, exist_ok=True)


def optimizeLogisticRegressionParameters(X_train, y_train):
    """
    @brief Trains a Logistic Regression model on the training data.

    This function trains a Logistic Regression model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Logistic Regression model.
    """
    param_grid = {
        'penalty': ['l1', 'l2', 'elasticnet'],
        'tol': [0.0001, 0.001, 0.01],
        'C': [0.1, 1.0, 10.0],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'multi_class': ['auto', 'ovr', 'multinomial'],
    }

    return optimizeModelParameters(LogisticRegression, 'Logistic Regression', param_grid, __LOGISTIC_RESULTS_PATH, X_train, y_train)


def trainAndTestLogisticRegression(X_train, y_train, X_test):
    """
    @brief Trains a Logistic Regression model on the training data.

    This function trains a Logistic Regression model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Logistic Regression and the predictions.
    """
    logging.info("Training Logistic Regression")
    return trainAndTestModel(LogisticRegression, X_train, y_train, X_test, __LOGISTIC_RESULTS_PATH)
